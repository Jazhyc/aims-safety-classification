import os
import json
import re
import gc
import shutil
import torch
from pathlib import Path
from tqdm import tqdm

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)


def _load_causal_lm(model_name: str, **kwargs):
    """Load a causal LM, routing Gemma 3 models to the text-only causal class.

    The HF repos for ``google/gemma-3-*-it`` are multimodal: the checkpoint stores
    weights under a ``language_model.*`` / ``vision_tower.*`` / ``multi_modal_projector.*``
    namespace, and ``AutoModelForCausalLM`` returns ``Gemma3ForConditionalGeneration``
    whose ``forward`` is vision-aware (it requires ``token_type_ids``).

    Loading directly with ``Gemma3ForCausalLM.from_pretrained`` doesn't work either,
    because its expected keys (``model.layers.*``) don't match the multimodal prefix
    so every weight ends up randomly initialized.

    Workaround: load the full multimodal model, then build a ``Gemma3ForCausalLM``
    wrapper that reuses the loaded language tower and lm_head, and drop the vision
    components to free memory.
    """
    cfg = AutoConfig.from_pretrained(model_name)
    if type(cfg).__name__ == "Gemma3Config":
        from transformers import Gemma3ForCausalLM, Gemma3ForConditionalGeneration

        full = Gemma3ForConditionalGeneration.from_pretrained(model_name, **kwargs)

        # Build a CausalLM wrapper on the meta device so __init__ runs
        # (registering submodules, generation_config, etc.) without allocating
        # any real parameters — we'll graft the already-loaded language tower
        # and lm_head from `full`.
        with torch.device("meta"):
            causal_lm = Gemma3ForCausalLM(full.config.text_config)
        causal_lm.generation_config = getattr(full, "generation_config", causal_lm.generation_config)
        causal_lm.model = full.model.language_model
        causal_lm.lm_head = full.lm_head
        # Signal that vLLM serves this model as *ForConditionalGeneration where
        # the language model lives under a `language_model.` prefix.  The LoRA
        # adapter keys saved by PEFT will use `model.layers.*`, but vLLM needs
        # `language_model.model.layers.*`.  The save path checks this flag and
        # renames the safetensors keys accordingly.
        causal_lm._vllm_lora_prefix = "language_model."

        # Free vision components — they're never used during text-only SFT.
        del full.model.vision_tower
        del full.model.multi_modal_projector
        del full
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return causal_lm

    if type(cfg).__name__ == "Mistral3Config":
        from transformers import Ministral3ForCausalLM, Mistral3ForConditionalGeneration

        full = Mistral3ForConditionalGeneration.from_pretrained(model_name, **kwargs)

        with torch.device("meta"):
            causal_lm = Ministral3ForCausalLM(full.config.text_config)
        causal_lm.generation_config = getattr(full, "generation_config", causal_lm.generation_config)
        causal_lm.model = full.model.language_model
        causal_lm.lm_head = full.lm_head
        causal_lm._vllm_lora_prefix = "language_model."

        del full.model.vision_tower
        del full.model.multi_modal_projector
        del full
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return causal_lm

    if type(cfg).__name__ == "Gemma4Config":
        from transformers import Gemma4ForCausalLM, Gemma4ForConditionalGeneration

        # Gemma 4 has global attention layers with head_dim=512, which exceeds
        # flash-attn's 256 limit. Fall back to eager — sdpa does not correctly
        # handle Gemma 4's mixed local/global attention with sequence packing.
        kwargs["attn_implementation"] = "eager"
        full = Gemma4ForConditionalGeneration.from_pretrained(model_name, **kwargs)

        # Gemma4ForCausalLM expects Gemma4TextConfig; build it on the meta device
        # so __init__ registers submodules without allocating parameters, then
        # graft the already-loaded language tower and lm_head from `full`.
        with torch.device("meta"):
            causal_lm = Gemma4ForCausalLM(full.config.text_config)
        causal_lm.generation_config = getattr(full, "generation_config", causal_lm.generation_config)
        causal_lm.model = full.model.language_model
        causal_lm.lm_head = full.lm_head
        causal_lm._vllm_lora_prefix = "language_model."

        # Free vision/audio components — never used during text-only SFT.
        for attr in ("vision_tower", "embed_vision", "audio_tower", "embed_audio"):
            if hasattr(full.model, attr):
                delattr(full.model, attr)
        del full
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return causal_lm

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    # Qwen3.5 (and potentially other future VLMs loaded via AutoModelForCausalLM)
    # may still be resolved as *ForConditionalGeneration with a language_model sub-module.
    if hasattr(model, "language_model") and not hasattr(model, "model"):
        model._vllm_lora_prefix = "language_model."
    return model


def _fix_lora_keys_for_vllm(adapter_dir: str, model) -> None:
    """Rename LoRA safetensors keys so they match vLLM's module paths.

    When training on a VLM (Gemma3, Mistral3, Qwen3.5, ...) we extract the
    language model tower into a CausalLM wrapper so PEFT can train on it.
    PEFT saves keys relative to that wrapper, e.g.:
        base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight

    But vLLM serves the full *ForConditionalGeneration model, where the same
    module is at:
        language_model.model.layers.0.self_attn.q_proj

    After stripping the `base_model.model.` prefix, vLLM would look for
    `model.layers.0...` and find nothing — silently running the base model.

    This function rewrites the adapter safetensors to add the missing prefix
    (`language_model.` by default) so vLLM finds the correct modules.
    """
    prefix = getattr(model, "_vllm_lora_prefix", None)
    if prefix is None:
        return  # Pure CausalLM (e.g. Llama) — no fix needed.

    sf_path = Path(adapter_dir) / "adapter_model.safetensors"
    if not sf_path.exists():
        return

    from safetensors import safe_open
    from safetensors.torch import save_file

    tensors = {}
    with safe_open(str(sf_path), framework="pt") as f:
        for key in f.keys():
            # base_model.model.<rest>  →  base_model.model.<prefix><rest>
            if key.startswith("base_model.model."):
                rest = key[len("base_model.model."):]
                new_key = f"base_model.model.{prefix}{rest}"
            else:
                new_key = key
            tensors[new_key] = f.get_tensor(key)

    save_file(tensors, str(sf_path))
    print(f"  [lora-fix] Renamed {len(tensors)} adapter keys: "
          f"base_model.model.<x> → base_model.model.{prefix}<x>")


from transformers.integrations import WandbCallback


class _WandbCallbackNoFinish(WandbCallback):
    """Prevent the Trainer from closing the W&B run so post-training inference
    metrics can be logged to the same run."""
    def on_train_end(self, args, state, control, **kwargs):
        pass
from trl import SFTTrainer, SFTConfig
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from .preprocessing import preprocess_data
from .data_utils import align_tokenizer_with_model, apply_binary_harm_mapping
from .evaluate_generations import compute_and_log_metrics
from .prompt_templates import build_student_prompt, build_student_messages
from .artifacts import artifact_exists, upload_adapter

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
except ImportError:
    LoraConfig = None
    get_peft_model = None
    prepare_model_for_kbit_training = None

def format_prompt(prompt_text, predict_harm=False):
    """
    Format the input prompt for the model.
    
    Args:
        prompt_text: The prompt text (raw text from dataset)
        predict_harm: Whether to include harm prediction task (unused, kept for compatibility)
    
    Returns:
        Formatted prompt string
    """
    return f"{prompt_text}\n"


def format_completion(intent, harm=None, predict_harm=False, classification_only=False):
    """
    Format the completion/target for the model.
    
    Args:
        intent: The intent text
        harm: The harm category (optional)
        predict_harm: Whether to include harm in completion
        classification_only: If True, only output harm label (no intent)
    
    Returns:
        Formatted completion string
    
    Examples:
        - classification_only=True, harm="harmful" -> "harmful"
        - predict_harm=True, intent="get info", harm="safe" -> "Intent: get info; Harm: safe"
        - predict_harm=False -> just the intent
    """
    if classification_only and harm:
        return harm
    
    if predict_harm and harm:
        return f"Intent: {intent}; Harm: {harm}"
    else:
        return intent


def format_completion_with_reasoning(reasoning, harm, intent=None):
    """
    Format the training completion for reasoning trace mode.

    Pipeline: prompt -> Reasoning: ... [-> Intent: ...] -> Prompt harm: ...

    Intent is included only when it is not None.  Harm is normalised from
    WildGuardMix format ("unharmful") to the binary format used by
    safety_experiment.py ("safe") so that downstream evaluation does not need
    a separate remapping step.

    Args:
        reasoning: Reasoning trace text generated for this sample.
        harm:      Ground-truth harm label ("harmful", "unharmful", or "safe").
        intent:    Human-written intent string; omit (None) for classification-only mode.

    Returns:
        Formatted completion string.
    """
    harm_normalised = "safe" if harm and harm.lower() == "unharmful" else (harm or "")
    parts = [f"Reasoning: {reasoning}"]
    if intent:
        parts.append(f"Intent: {intent}")
    parts.append(f"Prompt harm: {harm_normalised}")
    return "\n".join(parts)


def parse_reasoning_output(raw_text, with_intent=False):
    """
    Parse model output produced in reasoning trace format.

    Expected format (same as the training completions above):
        Reasoning: <step-by-step analysis>
        [Intent: <inferred intent>]       <- only in generation mode
        Prompt harm: <harmful|safe>

    Args:
        raw_text:    Raw string generated by the model.
        with_intent: Whether to extract an Intent field.

    Returns:
        (reasoning, intent, predicted_harm)
        intent is None when with_intent=False.
        predicted_harm is "harmful" or "safe" (None if unparseable).
    """
    text = raw_text.strip()

    # Extract Reasoning (up to Intent:, Prompt harm:, or end of string)
    reasoning = ""
    reasoning_match = re.search(
        r'Reasoning:\s*(.+?)(?=Intent:|Prompt harm:|$)', text, re.IGNORECASE | re.DOTALL
    )
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()

    # Extract Intent (optional)
    intent = None
    if with_intent:
        intent_match = re.search(
            r'Intent:\s*(.+?)(?=Prompt harm:|$)', text, re.IGNORECASE | re.DOTALL
        )
        if intent_match:
            intent = intent_match.group(1).strip()

    # Extract Prompt harm label
    predicted_harm = None
    harm_match = re.search(r'Prompt harm:\s*(harmful|unharmful|safe)', text, re.IGNORECASE)
    if harm_match:
        raw_harm = harm_match.group(1).lower()
        predicted_harm = "safe" if raw_harm == "unharmful" else raw_harm

    return reasoning, intent, predicted_harm


def _normalise_harm(label: str) -> str:
    """Map a predicted harm string to 'harmful' or 'safe'."""
    label = label.strip().lower()
    if "unharmful" in label or "safe" in label:
        return "safe"
    if "harmful" in label:
        return "harmful"
    return label


def load_reasoning_traces_dataset(data_cfg):
    """
    Load reasoning traces from a parsed_results.json file produced by
    generate_reasoning_traces.py and return a HuggingFace Dataset suitable
    for SFT training.

    ``reasoning_traces_condition`` accepts either a string or a list/sequence
    (e.g. Hydra ListConfig). When a sequence is given, records whose
    ``condition`` is in the set are all kept, and the intent source is selected
    per-record from the record's own ``condition`` field (``predicted.prompt_intent``
    for ``synthetic_intent``, ``ground_truth.intent`` otherwise). This enables
    mixed-source training sets that combine multiple condition labels in a
    single corpus (e.g. annotated ``human_intent`` rows alongside broader-pool
    ``synthetic_intent`` rows).

    Only records with a non-empty reasoning trace AND a resolvable binary harm
    label are kept. The harm label is normalised to "harmful"/"safe".

    Required config keys (under ``data``):
        reasoning_traces_path       -- path to parsed_results.json
        reasoning_traces_condition  -- str OR list of str

    Returns:
        HuggingFace Dataset with columns: id, prompt, intent, harm_label, reasoning, condition
    """
    from collections import Counter
    from datasets import Dataset
    from .data_utils import map_harm_to_binary

    traces_path = data_cfg.get("reasoning_traces_path")
    condition_raw = data_cfg.get("reasoning_traces_condition", "without_intent")

    if isinstance(condition_raw, str):
        conditions = [condition_raw]
    else:
        conditions = [str(c) for c in condition_raw]
    condition_set = set(conditions)

    if not traces_path:
        raise ValueError(
            "data.reasoning_traces_path must be set when use_reasoning_traces=True"
        )

    print(f"Loading reasoning traces from: {traces_path}")
    if len(conditions) == 1:
        print(f"  Condition filter: {conditions[0]}")
    else:
        print(f"  Condition filter (mixed): {conditions}")

    filter_disagreements = data_cfg.get("filter_teacher_disagreements", False)

    with open(traces_path) as f:
        records = json.load(f)

    available_conditions = list({r.get("condition") for r in records})
    filtered = []
    n_dropped_harm = 0
    for rec in records:
        rec_cond = rec.get("condition")
        if rec_cond not in condition_set:
            continue
        reasoning = rec.get("reasoning", "").strip()
        harm_raw = rec.get("ground_truth", {}).get("prompt_harm_label")
        harm_binary = map_harm_to_binary(harm_raw)
        if not reasoning or harm_binary is None:
            continue

        if filter_disagreements:
            pred_harm = (rec.get("predicted") or {}).get("prompt_harm", "")
            if pred_harm and harm_binary != _normalise_harm(pred_harm):
                n_dropped_harm += 1
                continue

        # Per-record intent source: synthetic uses model-generated; everything else uses GT.
        if rec_cond == "synthetic_intent":
            intent = rec.get("predicted", {}).get("prompt_intent", "") or ""
        else:
            intent = rec.get("ground_truth", {}).get("intent", "")

        filtered.append({
            "id": str(rec.get("annotation_id") or rec.get("wildguard_id", "")),
            "prompt": rec.get("prompt", ""),
            "intent": intent,
            "harm_label": harm_binary,   # "harmful" or "safe"
            "reasoning": reasoning,
            "condition": rec_cond,
        })

    if not filtered:
        raise ValueError(
            f"No valid reasoning traces found for condition '{condition_raw}' in "
            f"{traces_path}.  Available conditions: {available_conditions}"
        )

    print(f"  Loaded {len(filtered)} examples (conditions={conditions})")
    if len(conditions) > 1:
        print(f"  Per-condition counts: {dict(Counter(r['condition'] for r in filtered))}")
    if filter_disagreements:
        print(f"  Filtered (harm disagreement):   {n_dropped_harm}")
    return Dataset.from_list(filtered)


def load_extra_train_data(path: str):
    """
    Load supplementary training examples from a JSONL file.

    Expected columns per record: prompt, intent, gold_harm.
    Returns a HuggingFace Dataset with columns (prompt, intent, Annotator Harm)
    so it can be passed directly through create_prompt_completion in causal.py.
    """
    import json
    from datasets import Dataset

    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                records.append({
                    "prompt":         r["prompt"],
                    "intent":         r["intent"],
                    "Annotator Harm": r["gold_harm"],
                })
    print(f"  Loaded {len(records)} extra training examples from {path}")
    return Dataset.from_list(records)


def save_preds_causal(model_name, llm, lora_request, tokenizer, eval_dataset, split_name, config, max_length=256):
    """
    Generate predictions using VLLM for faster inference.
    
    Args:
        model_name: Name of the model (for file naming)
        llm: Initialized VLLM LLM instance
        lora_request: Initialized VLLM LoRARequest instance (or None)
        tokenizer: Tokenizer for formatting prompts
        eval_dataset: Dataset to generate predictions on
        split_name: Name of the split (e.g., "val", "test")
        config: Configuration dictionary
        max_length: Maximum input length for tokenization
    """
    paths_cfg = config.get("paths", {})
    gen_cfg = config.get("generation", {})
    data_cfg = config.get("data", {})
    predict_harm = data_cfg.get("predict_harm", False)
    classification_only = data_cfg.get("classification_only", False)
    use_reasoning_traces = data_cfg.get("use_reasoning_traces", False)
    # with_intent is derived from the trace condition: generation mode when condition produces intent
    with_intent = data_cfg.get("reasoning_traces_condition", "no_intent") in ("with_intent", "synthetic_intent", "human_intent")

    # Use different directory based on mode
    if use_reasoning_traces:
        suffix = "generation" if with_intent else "classification"
        base_pred_dir = paths_cfg.get("predictions_dir", "predictions_causal")
        base_pred_dir = base_pred_dir.replace("predictions", f"predictions_reasoning_{suffix}")
    elif classification_only:
        base_pred_dir = paths_cfg.get("predictions_dir", "predictions_causal")
        base_pred_dir = base_pred_dir.replace("predictions", "predictions_classification")
    elif predict_harm:
        base_pred_dir = paths_cfg.get("predictions_dir", "predictions_causal")
        base_pred_dir = base_pred_dir.replace("predictions", "predictions_with_harm")
    else:
        base_pred_dir = paths_cfg.get("predictions_dir", "predictions_causal")
    
    os.makedirs(base_pred_dir, exist_ok=True)

    filename = f"{model_name.replace('/', '_')}_{split_name}.jsonl"
    full_path = os.path.join(base_pred_dir, filename)

    # Generation settings
    gen_max_new_tokens = int(gen_cfg.get("max_new_tokens", 32))
    num_beams = int(gen_cfg.get("num_beams", 1))
    do_sample = bool(gen_cfg.get("do_sample", False))
    
    # Sampling parameters
    temperature = float(gen_cfg.get("temperature", 1.0))
    top_p = float(gen_cfg.get("top_p", 1.0))
    top_k = int(gen_cfg.get("top_k", 50))
    
    # Build sampling parameters for VLLM
    sampling_params = SamplingParams(
        max_tokens=gen_max_new_tokens,
        temperature=temperature if do_sample else 0.0,
        top_p=top_p if do_sample else 1.0,
        top_k=top_k if do_sample else -1,
        skip_special_tokens=True,
    )
    
    # Prepare all prompts
    examples = list(eval_dataset)
    prompt_texts = [ex['prompt'] for ex in examples]
    
    # Collect all predictions and references for metric computation
    all_preds = []
    all_refs = []

    # Generate all predictions with VLLM (handles batching internally)
    print(f"Generating {split_name} predictions with VLLM...")
    outputs = llm.generate(prompt_texts, sampling_params, lora_request=lora_request)
    
    # Process and save results
    with open(full_path, "w", encoding="utf-8") as f:
        for ex, output in zip(examples, outputs):
            generated_text = output.outputs[0].text.strip()

            if use_reasoning_traces:
                # Reasoning traces mode: parse Reasoning / [Intent] / Prompt harm
                # Output format mirrors safety_experiment.py's zeroshot_cot_* conditions
                # so that the same analysis notebooks and evaluation scripts work.
                reasoning, generated_intent, predicted_harm = parse_reasoning_output(
                    generated_text, with_intent=with_intent
                )
                true_harm = ex.get("harm_label")   # already binary: "harmful"/"safe"
                condition_name = (
                    "finetuned_reasoning_generation"
                    if with_intent
                    else "finetuned_reasoning_classification"
                )
                json_line = {
                    "id": ex["id"],
                    "prompt": ex["prompt"],
                    "true_harm": true_harm,
                    "true_harm_binary": true_harm,
                    "predicted_harm": predicted_harm,
                    "reasoning": reasoning,
                    "raw_generation": generated_text,
                    "condition": condition_name,
                }
                if with_intent:
                    json_line["true_intent"] = ex.get("intent")
                    json_line["generated_intent"] = generated_intent
                all_refs.append(true_harm or "")
                all_preds.append(predicted_harm or "")
            else:
                # Existing modes
                harm = ex.get("Annotator Harm") if (predict_harm or classification_only) else None
                true_formatted = format_completion(
                    ex["intent"], harm, predict_harm, classification_only
                )
                all_preds.append(generated_text)
                all_refs.append(true_formatted)

                if classification_only:
                    json_line = {
                        "id": ex["id"],
                        "prompt": ex["prompt"],
                        "true_harm": harm,
                        "pred_harm": generated_text,
                    }
                else:
                    json_line = {
                        "id": ex["id"],
                        "prompt": ex["prompt"],
                        "true_intent": true_formatted,
                        "generated_intent": generated_text,
                    }
            
            f.write(json.dumps(json_line, ensure_ascii=False) + "\n")

    print(f"Saved {split_name} predictions to {full_path}")
    return all_refs, all_preds


def setup_causal_model_and_tokenizer(config):
    """
    Set up tokenizer and model for causal LM training.

    - If config['peft']['use_lora'] is False or missing:
        → standard full fine-tuning.
    - If True:
        → load model in 4-bit and wrap with LoRA (QLoRA-style).

    Returns:
        tokenizer, model, is_peft
    """
    model_cfg = config.get("model", {})
    peft_cfg = config.get("peft", {})
    quant_cfg = config.get("quantization", {})
    model_name = model_cfg["name"]

    use_lora = bool(peft_cfg.get("use_lora", False))
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = 'left'

    if not use_lora:
        attn_implementation = model_cfg.get("attn_implementation", "flash_attention_2")
        model = _load_causal_lm(
            model_name,
            attn_implementation=attn_implementation,
            dtype=torch.bfloat16,
        )
        model.config.use_cache = False
        align_tokenizer_with_model(tokenizer, model)
        return tokenizer, model, False

    # === QLoRA path ===
    if BitsAndBytesConfig is None or LoraConfig is None or get_peft_model is None:
        raise ImportError(
            "QLoRA mode requested but `bitsandbytes` and/or `peft` are not installed.\n"
            "Install them with: pip install bitsandbytes peft"
        )

    compute_dtype_str = quant_cfg.get("bnb_4bit_compute_dtype", "bfloat16")
    compute_dtype = getattr(torch, compute_dtype_str, torch.bfloat16)
    trust_remote_code = bool(model_cfg.get("trust_remote_code", False))
    attn_implementation = model_cfg.get("attn_implementation", "flash_attention_2")

    # Some models (e.g. openai/gpt-oss-20b) ship pre-quantized with their own
    # quantization_config (e.g. Mxfp4Config). Passing a BitsAndBytesConfig on top
    # raises a ValueError. Detect this and skip BnB for pre-quantized models.
    pre_cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model_already_quantized = getattr(pre_cfg, "quantization_config", None) is not None

    if model_already_quantized:
        print(f"Model {model_name} is pre-quantized; skipping BitsAndBytes config.")
        model = _load_causal_lm(
            model_name,
            device_map="auto",
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
        )
    else:
        # 4-bit QLoRA quantization
        load_in_4bit = bool(quant_cfg.get("load_in_4bit", True))
        bnb_4bit_use_double_quant = bool(quant_cfg.get("bnb_4bit_use_double_quant", True))
        bnb_4bit_quant_type = quant_cfg.get("bnb_4bit_quant_type", "nf4")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model = _load_causal_lm(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
            dtype=compute_dtype,
        )

    model.config.use_cache = False

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_r = int(peft_cfg.get("lora_rank", 16))
    lora_alpha = int(peft_cfg.get("lora_alpha", 32))
    lora_dropout = float(peft_cfg.get("lora_dropout", 0.05))
    bias = peft_cfg.get("bias", "none")

    default_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    target_modules = peft_cfg.get("target_modules", default_target_modules)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    align_tokenizer_with_model(tokenizer, model)

    model.print_trainable_parameters()

    return tokenizer, model, True


def prepare_training_arguments(config, is_peft=False, num_train_samples=None):
    """
    Extract and prepare training arguments from config.
    """
    train_cfg = config.get("training", {})
    paths_cfg = config.get("paths", {})
    model_cfg = config.get("model", {})
    
    model_name = model_cfg["name"]
    out_name = model_name.replace("/", "_")
    
    output_dir = paths_cfg.get("output_dir", f"./data/train_results/{out_name}")
    logs_dir = paths_cfg.get("logs_dir", f"./logs/{out_name}")
    
    epochs = train_cfg.get("epochs", 8)
    lr = train_cfg.get("learning_rate", 5e-5)
    batch_size = train_cfg.get("batch_size", 8)
    weight_decay = train_cfg.get("weight_decay", 0.01)
    grad_accum = train_cfg.get("gradient_accumulation", 1)
    
    use_fp16 = train_cfg.get("fp16", False) and torch.cuda.is_available()
    use_bf16 = train_cfg.get("bf16", False) and torch.cuda.is_available()
    
    if use_fp16 and use_bf16:
        use_fp16 = False
    
    default_optim = "paged_adamw_8bit"
    optim_name = train_cfg.get("optim", default_optim)
    
    adam_beta1 = train_cfg.get("adam_beta1", None)
    adam_beta2 = train_cfg.get("adam_beta2", None)
    
    torch_compile = train_cfg.get("torch_compile", True)
    gradient_checkpointing = train_cfg.get("gradient_checkpointing", True)
    lr_scheduler_type = train_cfg.get("lr_scheduler_type", "cosine")
    warmup_ratio = train_cfg.get("warmup_ratio", 0.1)
    padding_free = train_cfg.get("padding_free", False)
    # TRL sequence packing — combined with padding_free, flash-attn varlen is
    # used so packed examples do not cross-attend (each gets its own causal
    # block via cumulative seqlens). Off by default to match the production
    # configs; flip on per-run for the data-scaling experiment after verifying
    # via scripts/hpc/scaling/verify_packing.py.
    packing = train_cfg.get("packing", False)
    max_length = model_cfg.get("max_length_causal", 512)
    
    wandb_cfg = config.get("wandb", {})
    report_to = "wandb" if wandb_cfg.get("enabled", False) else "none"
    run_name = wandb_cfg.get("run_name", None)
    
    training_args = {
        "output_dir": output_dir,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "learning_rate": lr,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "num_train_epochs": epochs,
        "weight_decay": weight_decay,
        "save_total_limit": 2,
        "save_only_model": True,
        "logging_dir": logs_dir,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "fp16": use_fp16,
        "bf16": use_bf16,
        "optim": optim_name,
        "torch_compile": torch_compile,
        "gradient_checkpointing": gradient_checkpointing,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "lr_scheduler_type": lr_scheduler_type,
        "warmup_ratio": warmup_ratio,
        "report_to": report_to,
        "run_name": run_name,
        "padding_free": padding_free,
        "packing": packing,
        "completion_only_loss": True,
        "remove_unused_columns": False,
        "max_length": max_length,
    }
    
    if adam_beta1 is not None:
        training_args["adam_beta1"] = adam_beta1
    if adam_beta2 is not None:
        training_args["adam_beta2"] = adam_beta2

    max_steps = train_cfg.get("max_steps", -1)
    if max_steps > 0:
        training_args["max_steps"] = max_steps

    return training_args


def run_causal_flow(config):
    model_cfg = config.get("model", {})
    paths_cfg = config.get("paths", {})
    train_cfg = config.get("training", {})
    model_name = model_cfg["name"]
    max_length = model_cfg.get("max_length_causal", 256)
    
    skip_training = bool(train_cfg.get("skip_training", False))

    # Artifact registry pre-flight check — fail fast before any GPU allocation
    artifacts_cfg = config.get("artifacts", {})
    artifacts_enabled = artifacts_cfg.get("enabled", False)
    registry_project = artifacts_cfg.get("registry_project", None)
    artifact_entity = artifacts_cfg.get("entity", None)

    if artifacts_enabled and not skip_training:
        _model_save_dir = paths_cfg.get("model_save_dir", f"./models/sft/{model_name}-model")
        _adapter_name = Path(_model_save_dir + "_adapter").name
        if artifact_exists(_adapter_name, registry_project, entity=artifact_entity):
            raise RuntimeError(
                f"Artifact '{_adapter_name}' already exists in W&B registry project "
                f"'{registry_project}'. Delete it from W&B before re-training, or "
                f"point paths.model_save_dir at a different directory."
            )

    # Get data configuration
    data_cfg = config.get("data", {})
    predict_harm = data_cfg.get("predict_harm", False)
    binary_harm_mapping = data_cfg.get("binary_harm_mapping", True)
    classification_only = data_cfg.get("classification_only", False)
    use_reasoning_traces = data_cfg.get("use_reasoning_traces", False)
    # rt_condition_raw is the reasoning traces condition spec — string or list of strings.
    rt_condition_raw = data_cfg.get("reasoning_traces_condition", "no_intent")
    if isinstance(rt_condition_raw, str):
        rt_conditions = [rt_condition_raw]
    else:
        rt_conditions = [str(c) for c in rt_condition_raw]

    # Decide template family. synthetic_intent and human_intent share the same
    # student template (PREAMBLE + OUTPUT_FORMAT_WITH_INTENT), so they can be
    # mixed in a single training set; no_intent is incompatible with both.
    _intent_family = {"synthetic_intent", "human_intent"}
    if all(c in _intent_family for c in rt_conditions):
        rt_template_condition = "synthetic_intent"
        with_intent = True
    elif all(c == "no_intent" for c in rt_conditions):
        rt_template_condition = "no_intent"
        with_intent = False
    elif rt_conditions == ["without_intent"]:  # legacy alias
        rt_template_condition = "no_intent"
        with_intent = False
    elif rt_conditions == ["with_intent"]:  # legacy alias
        rt_template_condition = "synthetic_intent"
        with_intent = True
    else:
        raise ValueError(
            f"reasoning_traces_condition={rt_conditions!r} mixes template families. "
            f"All values must be intent-producing ({sorted(_intent_family)}) or all 'no_intent'."
        )

    # Stable label for printing and val_metrics.json; "+" sentinel preserves order.
    rt_condition_label = rt_conditions[0] if len(rt_conditions) == 1 else "+".join(rt_conditions)

    # Log the training mode
    if use_reasoning_traces:
        print("=" * 60)
        if with_intent:
            print("REASONING TRACES MODE (generation: reasoning + intent + harm)")
            print("Training format: prompt -> Reasoning: ... / Intent: ... / Prompt harm: ...")
        else:
            print("REASONING TRACES MODE (classification: reasoning + harm)")
            print("Training format: prompt -> Reasoning: ... / Prompt harm: ...")
        print(f"Source condition: {rt_condition_label}")
        print(f"Traces path:      {data_cfg.get('reasoning_traces_path', '(not set)')}")
        print("=" * 60)
    elif classification_only:
        print("=" * 60)
        print("CLASSIFICATION-ONLY MODE")
        print("Training format: prompt -> harm label (e.g., 'harmful' or 'safe')")
        print("=" * 60)
    elif predict_harm:
        print("=" * 60)
        print("GENERATION MODE (with harm)")
        print("Training format: prompt -> Intent: {intent}; Harm: {harm}")
        print("=" * 60)
    else:
        print("=" * 60)
        print("GENERATION MODE (intent only)")
        print("Training format: prompt -> intent")
        print("=" * 60)

    if use_reasoning_traces:
        # Load pre-generated reasoning traces as training data.
        # Each example already has a reasoning field and a binary harm_label;
        # intent is available for the "with_intent" condition.

        # Load tokenizer early so create_prompt_completion can apply the chat template.
        # setup_causal_model_and_tokenizer will load it again (from cache) for training.
        _early_tok = AutoTokenizer.from_pretrained(model_name)

        def _apply_template(messages):
            try:
                return _early_tok.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                return _early_tok.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

        def create_prompt_completion(examples):
            prompts = [
                _apply_template(build_student_messages(p, condition=rt_template_condition))
                for p in examples["prompt"]
            ]
            intents = examples["intent"] if with_intent else [None] * len(examples["prompt"])
            completions = [
                format_completion_with_reasoning(r, h, i)
                for r, h, i in zip(examples["reasoning"], examples["harm_label"], intents)
            ]
            return {"prompt": prompts, "completion": completions}

        raw_train = load_reasoning_traces_dataset(data_cfg)
        train_dataset = raw_train.map(create_prompt_completion, batched=True)

        val_traces_path = data_cfg.get("reasoning_traces_val_path")
        test_traces_path = data_cfg.get("reasoning_traces_test_path")
        if val_traces_path:
            val_data_cfg = {**data_cfg, "reasoning_traces_path": val_traces_path}
            raw_val = load_reasoning_traces_dataset(val_data_cfg)
            if test_traces_path:
                from datasets import concatenate_datasets
                test_data_cfg = {**data_cfg, "reasoning_traces_path": test_traces_path}
                raw_test = load_reasoning_traces_dataset(test_data_cfg)
                raw_val = concatenate_datasets([raw_val, raw_test])
                print(f"  Combined val+test traces: {len(raw_val)} examples")
        else:
            print("WARNING: data.reasoning_traces_val_path not set — falling back to train split for validation")
            raw_val = raw_train
        val_dataset = raw_val.map(create_prompt_completion, batched=True)

        print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    else:
        def create_prompt_completion(examples):
            prompts = [format_prompt(p, predict_harm) for p in examples["prompt"]]

            if predict_harm or classification_only:
                harms = examples.get("Annotator Harm", [None] * len(examples["prompt"]))
                completions = [
                    format_completion(i, h, predict_harm, classification_only)
                    for i, h in zip(examples["intent"], harms)
                ]
            else:
                completions = [
                    format_completion(i, None, predict_harm, classification_only)
                    for i in examples["intent"]
                ]

            return {"prompt": prompts, "completion": completions}

        def _load_and_format(split):
            ds = preprocess_data(split=split)
            ds = apply_binary_harm_mapping(ds, binary_harm_mapping)
            return ds.map(create_prompt_completion, batched=True)

        print("Loading dataset splits from Hub...")
        train_dataset = _load_and_format("train")
        from datasets import concatenate_datasets
        val_dataset = concatenate_datasets([_load_and_format("validation"), _load_and_format("test")])
        print(f"  Combined val+test: {len(val_dataset)} examples")

        extra_train_path = data_cfg.get("extra_train_data")
        if extra_train_path:
            from datasets import concatenate_datasets
            extra_ds = load_extra_train_data(extra_train_path)
            extra_ds = apply_binary_harm_mapping(extra_ds, binary_harm_mapping)
            extra_ds = extra_ds.map(create_prompt_completion, batched=True)
            train_dataset = concatenate_datasets([train_dataset, extra_ds])
            print(f"  Augmented train with {len(extra_ds)} extra examples from {extra_train_path}")

        print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # Print example of training data
    print("\n" + "=" * 60)
    print("EXAMPLE TRAINING DATA:")
    print("=" * 60)
    example = train_dataset[0]
    print("── PROMPT ──")
    print(example["prompt"])
    print("── COMPLETION ──")
    print(example["completion"])
    print("=" * 60 + "\n")
    
    if skip_training:
        print("Skipping training (skip_training=True)")
        model_save_dir = paths_cfg.get(
            "model_save_dir",
            f"./models/sft/{model_name}-model",
        )
        print(f"Loading pre-trained model from {model_save_dir}")
        tokenizer = AutoTokenizer.from_pretrained(model_save_dir)
    else:
        tokenizer, model, is_peft = setup_causal_model_and_tokenizer(config)

        if not is_peft:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

        training_args = SFTConfig(**prepare_training_arguments(config, is_peft, len(train_dataset)))

        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=1,
            early_stopping_threshold=0.0,
        )

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=tokenizer,
            callbacks=[early_stopping, _WandbCallbackNoFinish()],
        )
        trainer.remove_callback(WandbCallback)

        trainer.evaluate()
        trainer.train()

        model_save_dir = paths_cfg.get(
            "model_save_dir",
            f"./models/sft/{model_name}-model",
        )
        if is_peft:
            print("Saving LoRA adapter for VLLM...")
            adapter_dir = model_save_dir + "_adapter"
            os.makedirs(adapter_dir, exist_ok=True)
            trainer.save_model(adapter_dir)
            _fix_lora_keys_for_vllm(adapter_dir, model)
            print(f"LoRA adapter saved to {adapter_dir}")
            tokenizer.save_pretrained(adapter_dir)
            try:
                import wandb as _wandb
                if _wandb.run is not None:
                    _wandb.run.summary["adapter_path"] = os.path.abspath(adapter_dir)
                    _wandb.run.summary["classification_only"] = bool(
                        config.get("data", {}).get("classification_only", False)
                    )
            except Exception:
                pass
            if artifacts_enabled:
                upload_adapter(adapter_dir, registry_project, entity=artifact_entity)
        else:
            os.makedirs(model_save_dir, exist_ok=True)
            trainer.save_model(model_save_dir)
            tokenizer.save_pretrained(model_save_dir)
            print(f"Model and tokenizer saved to {model_save_dir}")

        # Delete the Trainer checkpoint directory — checkpoints are only needed
        # during training for load_best_model_at_end; the final adapter above is
        # the only artifact we keep.
        checkpoint_dir = trainer.args.output_dir
        if os.path.isdir(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
            print(f"Cleaned up training checkpoints: {checkpoint_dir}")

        eval_results = trainer.evaluate()
        val_eval_loss = eval_results["eval_loss"]
        print(f"[Causal LM] Validation loss: {val_eval_loss}")

        del model
        del trainer
        del train_dataset
        
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            print(f"GPU Memory after cleanup: {allocated:.2f} GiB allocated, {reserved:.2f} GiB reserved")
        
        print("Freed PyTorch model from GPU memory")

    skip_vllm_eval = bool(train_cfg.get("skip_vllm_eval", False))
    if skip_vllm_eval:
        print("Skipping post-training vLLM evaluation (skip_vllm_eval=True)")
        return

    # Determine paths for VLLM loading
    if skip_training:
        adapter_dir = model_save_dir + "_adapter"
        if os.path.exists(adapter_dir):
            base_model_path = model_name
            adapter_path = adapter_dir
            print(f"Using LoRA adapter from {adapter_path} with base model {base_model_path}")
        else:
            base_model_path = model_save_dir
            adapter_path = None
            print(f"Using full model from {base_model_path}")
    else:
        if is_peft:
            base_model_path = model_name
            adapter_path = model_save_dir + "_adapter"
            print(f"Using LoRA adapter from {adapter_path} with base model {base_model_path}")
        else:
            base_model_path = model_save_dir
            adapter_path = None
            print(f"Using full model from {base_model_path}")

    # Initialize VLLM once
    peft_cfg = config.get("peft", {})
    if adapter_path:
        max_lora_rank = peft_cfg.get("lora_rank", 16)
        print(f"Loading model with VLLM and LoRA from {base_model_path} and {adapter_path}...")
        llm = LLM(
            model=base_model_path, 
            enable_lora=True, 
            max_lora_rank=max_lora_rank, 
            max_loras=1,
            limit_mm_per_prompt={"image": 0},
            gpu_memory_utilization=0.90,
            max_model_len=2048,
            dtype="bfloat16",
            enforce_eager=True
        )
        lora_request = LoRARequest("intent_lora", 1, adapter_path)
    else:
        print(f"Loading model with VLLM from {base_model_path}...")
        llm = LLM(
            model=base_model_path, 
            limit_mm_per_prompt={"image": 0},
            gpu_memory_utilization=0.90,
            max_model_len=2048,
            dtype="bfloat16",
            enforce_eager=True
        )
        lora_request = None

    # Generate predictions using VLLM
    val_refs, val_preds = save_preds_causal(
        model_name, llm, lora_request, tokenizer, val_dataset, "val", config,
        max_length=max_length,
    )

    del llm

    # Compute and log metrics
    metrics = compute_and_log_metrics(val_refs, val_preds, split_name="val")

    # Save metrics locally alongside the adapter / model weights
    save_dir = adapter_path or base_model_path
    if not skip_training:
        metrics["val_eval_loss"] = val_eval_loss
    metrics["model"] = model_name
    if use_reasoning_traces:
        metrics["condition"] = rt_condition_label
    metrics["learning_rate"] = train_cfg.get("learning_rate")
    metrics_path = os.path.join(save_dir, "val_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Validation metrics saved to {metrics_path}")
