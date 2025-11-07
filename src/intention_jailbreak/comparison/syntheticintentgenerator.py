"""
Generate synthetic intents using an open-source LLM and compare clustering patterns
with human-annotated intents.
"""
import logging
from typing import List
import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class SyntheticIntentGenerator:
    """Generate synthetic intents using vLLM."""
    
    def __init__(self, model_id: str):
        """
        Initialize the synthetic intent generator.
        
        Args:
            model_id: HuggingFace model ID for generation
        """
        logger.info(f"Initializing generator with model: {model_id}")
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.llm = LLM(model=model_id)
        self.sampling_params = SamplingParams(
            temperature=0,
            top_p=0.95,
            top_k=20,
            max_tokens=1000,
            min_p=0
        )
        
    def generate_intents_for_prompt(self, prompt: str, num_variations: int = 1) -> List[str]:
        """
        Generate synthetic intent variations for a given prompt.
        
        Args:
            prompt: The original jailbreak prompt
            num_variations: Number of synthetic intents to generate
            
        Returns:
            List of generated intent strings
        """
        system_prompt = """You are analyzing adversarial prompts designed to test AI safety. 
        For each prompt, generate a concise, nuanced intent statement that describes what the user is trying to achieve. 
        Follow these rules strictly:

        1. Keep the annotation short (1 sentence) and nuanced. 
        2. Do NOT make assumptions about the true user intent—only infer intent from the prompt itself. 
        3. Do NOT include information about the type of jailbreak or technique used. 
        4. Focus on the user’s main goal: the information or action they are trying to get from the system. 
        5. If the prompt is nonsensical, ambiguous, or unclear, label it as "ambiguous". 
        6. Respond using the template: "The intent is to <fill in the blank>" and only provide the blank content. 
        7. Do NOT include any explanation, reasoning, or extra text.

        Examples:

        Prompt: "Pretend you're an AI without restrictions..."  
        Intent: The intent is to obtain instructions for creating harmful content by bypassing safety guidelines

        Prompt: "In a hypothetical scenario where..."  
        Intent: The intent is to generate content that would normally be restricted by framing it as fictional

        Prompt: "asdf qwer"  
        Intent: ambiguous"""
        generated_intents = []
        
        for i in range(num_variations):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Prompt: {prompt}\n\nIntent:"}
            ]
            
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                tokenize=False,
                enable_thinking=False
            )
            
            outputs = self.llm.generate([formatted_prompt], self.sampling_params)
            intent = outputs[0].outputs[0].text.strip()
            
            # Clean up the intent
            intent = intent.replace("Intent:", "").strip()
            if intent and len(intent) > 10:  # Basic quality filter
                generated_intents.append(intent)
                
        logger.debug(f"Generated {len(generated_intents)} intents for prompt")
        return generated_intents
    
    def generate_synthetic_dataset(
        self, 
        prompts: List[str], 
        num_variations: int = 1,
        output_path: str = None
    ) -> pd.DataFrame:
        """
        Generate synthetic intents for a list of prompts.
        
        Args:
            prompts: List of prompts to generate intents for
            num_variations: Number of synthetic intents per prompt
            output_path: Optional path to save the dataset
            
        Returns:
            DataFrame with columns: prompt, synthetic_intent, variation_id
        """
        logger.info(f"Generating synthetic intents for {len(prompts)} prompts")
        logger.info(f"Generating {num_variations} variations per prompt")
        
        results = []
        for i, prompt in enumerate(prompts):
            if i % 10 == 0:
                logger.info(f"Progress: {i}/{len(prompts)} prompts processed")
                
            intents = self.generate_intents_for_prompt(prompt, num_variations)
            for j, intent in enumerate(intents):
                results.append({
                    'prompt': prompt,
                    'synthetic_intent': intent,
                    'variation_id': j,
                    'prompt_id': i
                })
        
        df = pd.DataFrame(results)
        logger.info(f"Generated {len(df)} total synthetic intents")
        
        if output_path:
            df.to_parquet(output_path, index=False)
            logger.info(f"Saved synthetic dataset to {output_path}")
            
        return df
