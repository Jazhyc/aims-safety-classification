import json
import argparse
from pathlib import Path


def detect_pred_key(example):
    """
    Decide which key contains the model prediction.

    - Seq2seq (T5-style): 'prediction'
    - Causal (Qwen/Llama): 'generated_intent'
    """
    if "prediction" in example:
        return "prediction"
    if "generated_intent" in example:
        return "generated_intent"
    raise KeyError(
        "Could not detect prediction key. Expected 'prediction' (seq2seq) "
        "or 'generated_intent' (causal). Found keys: "
        f"{list(example.keys())}"
    )


def normalize_text(text):
    """Basic normalization; expand later if needed."""
    if text is None:
        return ""
    return str(text).strip()


def prepare_file(input_path: Path, output_path: Path):
    """
    Read a predictions JSONL file and write a new JSONL with:
        { "id": ..., "true": "...", "pred": "..." }
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Reading predictions from: {input_path}")
    num_lines = 0
    num_written = 0

    with input_path.open("r", encoding="utf-8") as fin:
        # Peek at the first non-empty line to auto-detect pred key
        first_example = None
        pos = fin.tell()
        for line in fin:
            line = line.strip()
            if not line:
                continue
            first_example = json.loads(line)
            break

        if first_example is None:
            raise ValueError("Input file appears to be empty.")

        pred_key = detect_pred_key(first_example)
        print(f"Detected prediction key: '{pred_key}'")

        # Rewind to the beginning to process all lines, including the first one
        fin.seek(pos)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fout:
            for line in fin:
                num_lines += 1
                line = line.strip()
                if not line:
                    continue

                ex = json.loads(line)

                # Required fields
                if "id" not in ex:
                    raise KeyError(f"Missing 'id' in example: {ex}")
                if "true_intent" not in ex:
                    raise KeyError(f"Missing 'true_intent' in example: {ex}")
                if pred_key not in ex:
                    raise KeyError(
                        f"Missing prediction key '{pred_key}' in example: {ex}"
                    )

                new_ex = {
                    "id": ex["id"],
                    "true": normalize_text(ex["true_intent"]),
                    "pred": normalize_text(ex[pred_key]),
                }

                fout.write(json.dumps(new_ex, ensure_ascii=False) + "\n")
                num_written += 1

    print(f"Wrote {num_written} examples to: {output_path}")
    if num_lines != num_written:
        print(
            f"Note: {num_lines} lines read, {num_written} non-empty examples written."
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare prediction files for evaluation by standardizing them to "
            "a common JSONL format: {id, true, pred}."
        )
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        type=str,
        help="Path to the input predictions JSONL file "
             "(e.g. t5-base_val.jsonl, Qwen*_val.jsonl).",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        type=str,
        help="Path to the output standardized JSONL file.",
    )

    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    prepare_file(input_path, output_path)


if __name__ == "__main__":
    main()
