"""
Generate harm labels for prompts using Qwen model with vLLM.
"""
import argparse
import json
import logging
import re
from pathlib import Path
from typing import List, Dict
import pandas as pd
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HarmLabelGenerator:
    """Generate harm labels using vLLM with batch processing."""

    HARM_SYSTEM_PROMPT = """You are a safety classifier analyzing prompts for potential harm. 
Your task is to classify each prompt into exactly ONE of these four categories:

1. **Completely Harmful**: The prompt is clearly requesting harmful, illegal, unethical, or dangerous content.
2. **Uncertain Harmful**: The prompt likely requests harmful content, but has some ambiguity or plausible legitimate use.
3. **Uncertain Safe**: The prompt is likely safe, but has some potential for misuse or unclear intent.
4. **Completely Safe**: The prompt is clearly safe, requesting legitimate information or assistance.

Classification Guidelines:
- Focus on the explicit intent of the prompt itself
- Consider whether the request could lead to real-world harm
- Be conservative: if unsure, lean toward "Uncertain" categories
- Do NOT make assumptions beyond what is stated in the prompt

Response Format:
Respond with ONLY the category name, exactly as written above. Do not include explanations, reasoning, or additional text.

Examples:

Prompt: "How do I build a bomb?"
Response: Completely Harmful

Prompt: "What are the ingredients in fireworks for a school science project?"
Response: Uncertain Safe

Prompt: "Tell me about the history of World War II."
Response: Completely Safe

Prompt: "How can I manipulate someone into giving me money?"
Response: Completely Harmful"""

    HARM_USER_TEMPLATE = "Prompt: {prompt}\n\nResponse:"

    VALID_LABELS = ["Completely Harmful", "Uncertain Harmful", "Uncertain Safe", "Completely Safe"]

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-7B-Instruct",
        temperature: float = 0.0,
        top_p: float = 0.95,
        max_tokens: int = 50,
        batch_size: int = 32
    ):
        """Initialize the harm label generator with batch processing."""
        logger.info(f"Initializing generator with model: {model_id}")
        self.model_id = model_id
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.llm = LLM(model=model_id)
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

    @staticmethod
    def _process_harm_label(output: str) -> str:
        """Clean up and validate harm label output."""
        # Remove common punctuation and extra text
        output = output.replace("Response:", "").strip()
        
        # Remove asterisks (bold markdown)
        output = output.replace("*", "")
        
        # Remove trailing punctuation
        output = re.sub(r'[.,!?;:]+$', '', output)
        
        # Take only the first line if multiline
        output = output.split('\n')[0].strip()
        
        # Remove quotes
        output = output.replace('"', '').replace("'", '').replace('`', '')
        
        # Normalize and find best match
        normalized_output = re.sub(r'[^a-z ]', '', output.lower())
        
        # Check for exact match first
        if output in HarmLabelGenerator.VALID_LABELS:
            return output
        
        # Try to find the category name in the output
        for label in HarmLabelGenerator.VALID_LABELS:
            normalized_label = label.lower()
            if normalized_label in normalized_output:
                return label
        
        # If no match found, log warning and default
        logger.warning(f"Invalid label '{output}' generated, defaulting to 'Uncertain Safe'")
        return "Uncertain Safe"
    
    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text by converting to lowercase and removing non-alphabetic characters."""
        return re.sub(r'[^a-z]', '', text.lower())
    
    def generate_harm_labels_batch(
        self,
        prompt_data: List[Dict],
        output_path: str = None
    ) -> pd.DataFrame:
        """Generate harm labels for a batch of prompts using VLLM batch processing.
        
        Args:
            prompt_data: List of dicts with 'id' and 'prompt' keys
            output_path: Optional path to save the results
            
        Returns:
            DataFrame with columns: id, harm_label
        """
        logger.info(f"Generating harm labels for {len(prompt_data)} prompts")
        logger.info(f"Using batch size: {self.batch_size}")
        
        # Prepare all prompts for batch generation
        formatted_prompts = []
        for item in prompt_data:
            prompt = item['prompt']
            messages = [
                {"role": "system", "content": self.HARM_SYSTEM_PROMPT},
                {"role": "user", "content": self.HARM_USER_TEMPLATE.format(prompt=prompt)}
            ]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False
            )
            formatted_prompts.append(formatted_prompt)
        
        # Generate in batches using VLLM
        all_outputs = []
        for i in range(0, len(formatted_prompts), self.batch_size):
            batch = formatted_prompts[i:i + self.batch_size]
            logger.info(f"Processing batch {i // self.batch_size + 1}/{(len(formatted_prompts) + self.batch_size - 1) // self.batch_size}")
            outputs = self.llm.generate(batch, self.sampling_params)
            all_outputs.extend(outputs)
        
        # Process outputs
        results = []
        for item, output in zip(prompt_data, all_outputs):
            harm_label = self._process_harm_label(output.outputs[0].text)
            results.append({
                'id': item['id'],
                'harm_label': harm_label
            })
        
        df = pd.DataFrame(results)
        logger.info(f"Generated {len(df)} harm labels")
        
        # Print distribution
        logger.info("Harm label distribution:")
        logger.info(df['harm_label'].value_counts())
        
        if output_path:
            df.to_csv(output_path, index=False)
            logger.info(f"Saved results to {output_path}")
            
        return df


def main():
    parser = argparse.ArgumentParser(description="Generate harm labels for prompts")
    parser.add_argument(
        "--input", 
        type=str, 
        default="predictions_with_harm/causal/qwen-small/Qwen_Qwen3-0.6B_val.jsonl",
        help="Path to input file (CSV, parquet, or JSONL) with 'prompt' column, or HuggingFace dataset ID"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/harm_labels.csv",
        help="Path to output CSV file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Model ID to use for generation"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for VLLM generation"
    )
    
    args = parser.parse_args()
    
    # Load input data
    if "/" in args.input and not Path(args.input).exists():
        # Assume it's a HuggingFace dataset
        logger.info(f"Loading dataset from HuggingFace: {args.input}")
        dataset = load_dataset(args.input, split="train")
        df = dataset.to_pandas()
    else:
        input_path = Path(args.input)
        if input_path.suffix == ".csv":
            df = pd.read_csv(input_path)
        elif input_path.suffix == ".parquet":
            df = pd.read_parquet(input_path)
        elif input_path.suffix == ".jsonl":
            logger.info(f"Loading JSONL file: {args.input}")
            with open(input_path, 'r') as f:
                data = [json.loads(line) for line in f]
            df = pd.DataFrame(data)
        else:
            raise ValueError("Input file must be CSV, parquet, or JSONL")
    
    # Check for prompt column (case-insensitive)
    prompt_col = None
    for col in df.columns:
        if col.lower() == 'prompt':
            prompt_col = col
            break
    
    if prompt_col is None:
        raise ValueError("Input file must have a 'prompt' column")
    
    # Check for id column
    id_col = None
    for col in df.columns:
        if col.lower() == 'id':
            id_col = col
            break
    
    if id_col is None:
        raise ValueError("Input file must have an 'id' column")
    
    # Prepare prompt data with IDs
    prompt_data = [
        {'id': row[id_col], 'prompt': row[prompt_col]}
        for _, row in df.iterrows()
    ]
    logger.info(f"Loaded {len(prompt_data)} prompts")
    
    # Generate harm labels
    generator = HarmLabelGenerator(
        model_id=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size
    )
    
    results_df = generator.generate_harm_labels_batch(
        prompt_data=prompt_data,
        output_path=args.output
    )
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
