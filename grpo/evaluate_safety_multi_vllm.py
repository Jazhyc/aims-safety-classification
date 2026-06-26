import os
import re
import argparse
import json
import pandas as pd
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from sklearn.metrics import f1_score, classification_report
from datasets import load_dataset
from transformers import AutoTokenizer

# [DATASET_CONFIGS remain exactly as in your original script]
DATASET_CONFIGS = {
    "annotated-intents": {
        "dataset_name": "Jazhyc/aims-safety-intents",
        "subset": None,
        "split": "validation+test",
        "harm_column": "Annotator Harm",
        "prompt_column": "Prompt",
        "intent_column": "Intent",
        "use_manual_split": False,
    },
    "validation_aegis": {
        "dataset_name": "nvidia/Aegis-AI-Content-Safety-Dataset-2.0",
        "subset": None,
        "split": "validation",
        "harm_column": "prompt_label",
        "prompt_column": "prompt",
        "use_manual_split": False,
    },
    "validation_toxic-chat": {
        "dataset_name": "lmsys/toxic-chat",
        "subset": "toxicchat0124",
        "split": "train",
        "harm_column": "toxicity",
        "prompt_column": "user_input",
        "use_manual_split": False,
        "label_map": {0: "safe", 1: "harmful"},
    },
    "wildguardmix": {
        "dataset_name": "allenai/wildguardmix",
        "subset": "wildguardtest",
        "split": "test",
        "harm_column": "prompt_harm_label",
        "prompt_column": "prompt",
        "use_manual_split": False,
    },
    "xstest": {
        "dataset_name": "walledai/XSTest",
        "subset": None,
        "split": "test",
        "harm_column": "label",
        "prompt_column": "prompt",
        "use_manual_split": False,
    },
    "toxic-chat": {
        "dataset_name": "lmsys/toxic-chat",
        "subset": "toxicchat0124",
        "split": "test",
        "harm_column": "toxicity",
        "prompt_column": "user_input",
        "use_manual_split": False,
        "label_map": {0: "safe", 1: "harmful"},
    },
    "aegis": {
        "dataset_name": "nvidia/Aegis-AI-Content-Safety-Dataset-2.0",
        "subset": None,
        "split": "test",
        "harm_column": "prompt_label",
        "prompt_column": "prompt",
        "use_manual_split": False,
    },
    "openai-moderation": {
        "dataset_name": "AllanK24/openai-moderation-binary",
        "subset": None,
        "split": "test",
        "harm_column": "prompt_label",
        "prompt_column": "prompt",
        "use_manual_split": False,
    },
}

# [Utility functions: opposite_label, extract_label, extract_intent, 
# get_intent_similarity, standardize_label, load_dataset_split, 
# prepare_dataset_for_evaluation remain UNCHANGED to preserve functionality]

def opposite_label(label):
    if label == 'safe': return 'harmful'
    elif label == 'harmful': return 'safe'
    raise ValueError(f'Invalid label {label}.')


def extract_label(text):
    m = re.search(r"Harm:\s*(harmful|safe)", text, re.IGNORECASE)
    pred = m.group(1).lower() if m else None
    if pred is None:
        print(f'Could not extract label from text: {text}\n')
    return pred

def extract_intent(solution_str):
    if solution_str is None: return None
    match = re.search(r'Intent:(.*?);', solution_str, re.DOTALL)
    return match.group(1).strip() if match else None

def get_intent_similarity(predicted_intent, golden_intent):
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    if predicted_intent is None or golden_intent is None: return 0.0
    if not hasattr(get_intent_similarity, "_model"):
        get_intent_similarity._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    model = get_intent_similarity._model
    embeddings = model.encode([predicted_intent, golden_intent])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return float(np.clip(similarity, 0.0, 1.0))

def standardize_label(label, label_map=None):
    if label_map: label = label_map.get(label, label)
    if isinstance(label, (int, float)): label = str(int(label))
    if isinstance(label, str): label = label.lower().strip()
    if label is None: return None
    if label.lower() in ['harmful', 'unsafe', '1', 'completely harmful', 'uncertain harmful']: return 'harmful'
    if label.lower() in ['unharmful', 'safe', '0', 'completely safe', 'uncertain safe']: return 'safe'
    raise NotImplementedError(f'Unrecognized label {label}')

def load_dataset_split(config):
    dataset_name, subset, split = config["dataset_name"], config.get("subset"), config["split"]
    if subset: dataset = load_dataset(dataset_name, subset, split=split)
    else: dataset = load_dataset(dataset_name, split=split)
    if "+" in split:
        splits = split.split("+")
        combined_data = []
        for s in splits:
            data = load_dataset(dataset_name, subset, split=s) if subset else load_dataset(dataset_name, split=s)
            combined_data.extend(data)
        return combined_data
    return list(dataset)

def prepare_dataset_for_evaluation(data, config, dataset_name):
    prepared_data = []
    harm_column, prompt_column = config["harm_column"], config["prompt_column"]
    intent_column, label_map = config.get("intent_column"), config.get("label_map")
    for item in data:
        if harm_column not in item or prompt_column not in item: continue
        prompt, raw_label = item[prompt_column], item[harm_column]
        std_label = standardize_label(raw_label, label_map)
        if std_label is not None:
            intent = item.get(intent_column, 'Unknown intent') if intent_column else 'N/A'
            prepared_data.append({
                'prompt': prompt,
                'reward_model': {'ground_truth': std_label},
                'extra_info': {'original_prompt': prompt, 'golden_intent': intent}
            })
    return prepared_data

def run_evaluation_on_dataset(llm, prepared_data, dataset_name, max_tokens, apply_chat_template, system_prompt, adapter_path=None):
    """vLLM-optimized evaluation loop."""
    print(f"\nEvaluating {dataset_name} ({len(prepared_data)} samples)...")
    
    y_true, y_pred, side_by_side_data = [], [], []
    tokenizer = llm.get_tokenizer()
    
    # Prepare all prompts at once for vLLM
    input_texts = []
    for item in prepared_data:
        prompt = item['prompt']
        if apply_chat_template:
            messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
            messages += [{"role": "user", "content": prompt}]
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            input_text = f"{prompt}\n"
        input_texts.append(input_text)

    # Setup sampling and LoRA if needed
    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    lora_request = LoRARequest("adapter", 1, adapter_path) if adapter_path else None

    # Run inference in one large batch (vLLM handles scheduling)
    outputs = llm.generate(input_texts, sampling_params, lora_request=lora_request)

    for i, output in enumerate(outputs):
        item = prepared_data[i]
        full_answer = output.outputs[0].text
        
        pred_label = extract_label(full_answer)
        pred_intent = extract_intent(full_answer)
        true_label = item['reward_model']['ground_truth']
        golden_intent = item['extra_info']['golden_intent']
        
        sim_score = 0.0
        if dataset_name == "annotated-intents" and golden_intent != 'N/A':
            sim_score = get_intent_similarity(pred_intent, golden_intent)
        
        side_by_side_data.append({
            "input_prompt": item['extra_info']['original_prompt'],
            "gold_intent": golden_intent,
            "gold_label": true_label,
            "full_answer": full_answer,
            "predicted_label": pred_label,
            "predicted_intent": pred_intent,
            "intent_similarity": sim_score,
            "is_correct": (pred_label == true_label),
            "input_text": input_texts[i]
        })

        y_true.append(true_label)
        y_pred.append(pred_label if pred_label else opposite_label(true_label))

    # Calculate Metrics
    report_dict = classification_report(y_true, y_pred, digits=4, zero_division=0, output_dict=True)
    f1_harmful = f1_score(y_true, y_pred, pos_label="harmful", labels=["safe", "harmful"], average='binary', zero_division=0)
    mean_sim = sum(d['intent_similarity'] for d in side_by_side_data) / len(side_by_side_data) if side_by_side_data else 0.0

    print(f"[{dataset_name.upper()}] F1 Harmful: {f1_harmful:.4f} | Mean Intent Sim: {mean_sim:.4f}")

    return pd.DataFrame(side_by_side_data), {
        "classification_report": report_dict,
        "f1_harmful": f1_harmful,
        "mean_intent_similarity": mean_sim,
        "total_samples": len(y_true)
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate a safety model using vLLM.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--apply_chat_template", action='store_true')
    parser.add_argument("--tp_size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--system_prompt_file", default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)

    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    # Initialize vLLM
    print(f"Loading model with vLLM from {args.model_path}...")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tp_size,
        dtype="bfloat16",
        # dtype="float32",
        enable_lora=(args.adapter_path is not None),
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=4096 # Adjust based on your model's capacity
    )

    if args.system_prompt_file:
        with open(args.system_prompt_file, 'r', encoding='utf-8') as f:
            system_prompt = f.read().strip()
    else:
        system_prompt = None

    datasets_to_evaluate = args.datasets if args.datasets else list(DATASET_CONFIGS.keys())
    all_metrics = {}

    for dataset_name in datasets_to_evaluate:
        if dataset_name not in DATASET_CONFIGS: continue
        config = DATASET_CONFIGS[dataset_name]
        
        try:
            raw_data = load_dataset_split(config)
            prepared_data = prepare_dataset_for_evaluation(raw_data, config, dataset_name)
            
            results_df, metrics = run_evaluation_on_dataset(
                llm, prepared_data, dataset_name, args.max_new_tokens, 
                args.apply_chat_template, system_prompt, args.adapter_path
            )
            
            # Save results
            dataset_out = os.path.join(args.output_path, dataset_name)
            os.makedirs(dataset_out, exist_ok=True)
            results_df.to_csv(os.path.join(dataset_out, f'predictions_{dataset_name}.csv'), index=False)
            with open(os.path.join(dataset_out, f'metrics_{dataset_name}.json'), 'w') as f:
                json.dump({dataset_name: metrics}, f, indent=4)
            
            all_metrics[dataset_name] = metrics
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            raise e

    # --- COMPUTE COMBINED METRICS ---

    internal_f1_scores = []
    for dataset_name, metrics in all_metrics.items():
        if dataset_name == "annotated-intents":
            internal_f1_scores.append(metrics["f1_harmful"])

    external_f1_scores = []
    for dataset_name, metrics in all_metrics.items():
        if dataset_name != "annotated-intents" and 'validation' not in dataset_name:
            external_f1_scores.append(metrics["f1_harmful"])

    validation_f1_scores = []
    for dataset_name, metrics in all_metrics.items():
        if 'validation' in dataset_name:
            validation_f1_scores.append(metrics["f1_harmful"])
    
    
    # Calculate averages
    all_f1_scores = [metrics["f1_harmful"] for metrics in all_metrics.values()]
    avg_f1_all = sum(all_f1_scores) / len(all_f1_scores) if all_f1_scores else 0.0
    avg_f1_internal = sum(internal_f1_scores) / len(internal_f1_scores) if internal_f1_scores else 0.0
    avg_f1_external = sum(external_f1_scores) / len(external_f1_scores) if external_f1_scores else 0.0
    avg_f1_validation = sum(validation_f1_scores) / len(validation_f1_scores) if validation_f1_scores else 0.0
    
    # Create combined metrics
    combined_metrics = {
        "individual_datasets": all_metrics,
        "summary": {
            "avg_f1_harmful_all_datasets": avg_f1_all,
            "avg_f1_harmful_internal_datasets": avg_f1_internal,
            "avg_f1_harmful_validation_datasets": avg_f1_validation,
            "avg_f1_harmful_external_datasets": avg_f1_external,
            "total_datasets_evaluated": len(all_metrics),
            "internal_datasets_evaluated": len(internal_f1_scores),
            "validation_datasets_evaluated": len(validation_f1_scores),
            "external_datasets_evaluated": len(external_f1_scores),
        }
    }
    
    # Save combined metrics
    with open(os.path.join(args.output_path, 'combined_metrics.json'), 'w') as f:
        json.dump(combined_metrics, f, indent=4)
    
    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    for dataset_name, metrics in all_metrics.items():
        print(f"{dataset_name:20} | F1 Harmful: {metrics['f1_harmful']:.4f}")
    print(f"{'='*60}")
    print(f"{'Average (All)':20} | F1 Harmful: {avg_f1_all:.4f}")
    print(f"{'Average (Internal)':20} | F1 Harmful: {avg_f1_internal:.4f}")
    print(f"{'Average (Validation)':20} | F1 Harmful: {avg_f1_validation:.4f}")
    print(f"{'Average (External)':20} | F1 Harmful: {avg_f1_external:.4f}")
    print(f"{'='*60}")

    print(f"\nEvaluation Complete. All files saved to: {args.output_path}")

if __name__ == "__main__":
    main()