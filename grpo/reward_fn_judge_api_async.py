import argparse
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from transformers import AutoTokenizer
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

SCRIPT_DIR = Path(__file__).parent

engine = None
tokenizer = None
MODEL_PATH = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, tokenizer

    print(f"Initializing vLLM Engine with model: {MODEL_PATH}")
    engine_args = AsyncEngineArgs(
        model=MODEL_PATH,
        gpu_memory_utilization=0.85,
        tensor_parallel_size=2,
        dtype="bfloat16",
        enforce_eager=True
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    yield

    print("Shutting down engine...")


app = FastAPI(lifespan=lifespan)


class JudgeItem(BaseModel):
    user_prompt: str
    reference_intent: str
    generated_intent: str


class BatchJudgeRequest(BaseModel):
    items: List[JudgeItem]


def create_judge_prompt(item: JudgeItem) -> str:
    system_prompt = (SCRIPT_DIR / "system_prompt_judge.txt").read_text(encoding="utf-8").strip()
    user_message = (
        f"User prompt:\n{item.user_prompt}\n\n"
        f"Reference intent:\n{item.reference_intent}\n\n"
        f"Generated intent:\n{item.generated_intent}\n\n"
        f"Verdict:"
    )
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@app.post("/judge_batch")
async def judge_intent_batch(request: BatchJudgeRequest):
    sampling_params = SamplingParams(
        max_tokens=10,
        temperature=0.0,
        top_p=1.0,
        skip_special_tokens=True,
    )

    prompts = [create_judge_prompt(item) for item in request.items]

    tasks = [
        engine.generate(p, sampling_params, request_id=f"req_{i}")
        for i, p in enumerate(prompts)
    ]

    results = []
    for task in tasks:
        final_output = None
        async for request_output in task:
            final_output = request_output

        response = final_output.outputs[0].text.strip().lower()
        if "good_match" in response:
            verdict = "good_match"
        elif "decent_match" in response:
            verdict = "decent_match"
        else:
            verdict = "bad_match"
        results.append(verdict)

    return {"verdicts": results}


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-27b-it", help="HuggingFace model name or local path")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    MODEL_PATH = args.model

    uvicorn.run(app, host="0.0.0.0", port=args.port)
