import os
import json
import argparse

def get_value_by_path(data, path):
    """Navigates a dictionary using a string path."""
    try:
        keys = path.split('/')
        val = data
        for k in keys:
            val = val[k]
        return val
    except (KeyError, TypeError):
        return -1.0

def format_val(val):
    """Formats float to 3 decimal places."""
    try:
        return f"{float(val):.3f}"
    except (ValueError, TypeError):
        return "0.000"

def get_checkpoint_row(model_name, data, best_step, last_step, metric_label):
    """Returns the specific LaTeX-formatted string as a list of lines."""
    datasets = data.get("individual_datasets", {})
    summary = data.get("summary", {})

    # Mapping based on your requirements
    metrics = [
        format_val(datasets.get("annotated-intents", {}).get("f1_harmful", 0)),
        format_val(datasets.get("wildguardmix", {}).get("f1_harmful", 0)),
        format_val(datasets.get("xstest", {}).get("f1_harmful", 0)),
        format_val(datasets.get("aegis", {}).get("f1_harmful", 0)),
        format_val(datasets.get("toxic-chat", {}).get("f1_harmful", 0)),
        format_val(datasets.get("openai-moderation", {}).get("f1_harmful", 0)),
        format_val(summary.get("avg_f1_harmful_external_datasets", 0)),
    ]

    val_aegis = format_val(datasets.get("validation_aegis", {}).get("f1_harmful", 0))
    val_toxic = format_val(datasets.get("validation_toxic-chat", {}).get("f1_harmful", 0))
    val_avg = format_val(summary.get("avg_f1_harmful_validation_datasets", 0))

    lines = []
    lines.append(f"\n# BEST FOR: {metric_label}")
    lines.append(f"# {model_name}")
    
    row_parts = []
    for m in metrics:
        row_parts.append(f"  & {m}")
    
    row_parts.append(f"  & 1")
    # Updated to show best_step/last_step
    row_parts.append(f"  & {best_step}/{last_step}")
    row_parts.append(f"  & {val_aegis}")
    row_parts.append(f"  & {val_toxic}")
    row_parts.append(f"  & {val_avg}")
    
    lines.extend(row_parts)
    lines.append(f"  \\\\")
    return lines

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folder", type=str, required=True, help="Path to the model root folder")
    args = parser.parse_args()

    model_name = os.path.basename(os.path.normpath(args.model_folder))
    output_file = os.path.join(args.model_folder, "best_checkpoints_report.txt")
    
    target_metrics = {
        "Internal Datasets": "summary/avg_f1_harmful_internal_datasets",
        "Validation Datasets": "summary/avg_f1_harmful_validation_datasets"
    }

    best_results = {
        "Internal Datasets": {"score": -1.0, "data": None, "step": "0"},
        "Validation Datasets": {"score": -1.0, "data": None, "step": "0"}
    }

    if not os.path.exists(args.model_folder):
        print(f"Folder {args.model_folder} not found.")
        return

    # 1. First pass: determine the last_step
    all_steps = []
    for folder in os.listdir(args.model_folder):
        if folder.startswith("results_step"):
            try:
                step_num = int(folder.replace("results_step", ""))
                all_steps.append(step_num)
            except ValueError:
                continue
    
    last_step = max(all_steps) if all_steps else "Unknown"

    # 2. Second pass: find the best checkpoints
    for folder in os.listdir(args.model_folder):
        if folder.startswith("results_step"):
            step_path = os.path.join(args.model_folder, folder, "combined_metrics.json")
            if os.path.exists(step_path):
                with open(step_path, 'r') as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        continue
                
                # Keep step as string for the report but it was found via the max() logic
                step_str = folder.replace("results_step", "")

                for label, path in target_metrics.items():
                    score = get_value_by_path(data, path)
                    if score > best_results[label]["score"]:
                        best_results[label]["score"] = score
                        best_results[label]["data"] = data
                        best_results[label]["step"] = step_str

    # 3. Write results to text file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Best Checkpoints Report for: {model_name}\n")
        f.write("="*50 + "\n")
        f.write(f"Last available step in folder: {last_step}\n")
        
        for label in target_metrics.keys():
            res = best_results[label]
            if res["data"]:
                row_lines = get_checkpoint_row(model_name, res["data"], res["step"], last_step, label)
                f.write("\n".join(row_lines) + "\n")
            else:
                f.write(f"\n# No data found for {label}\n")

    print(f"Results successfully saved to: {output_file}")

if __name__ == "__main__":
    main()