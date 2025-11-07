"""
Plot calibration curve from saved test predictions.

This script:
1. Loads test predictions with probabilities
2. Computes calibration metrics
3. Generates calibration curve plot
4. Optionally logs to wandb

Usage:
    python scripts/plot_calibration.py
    python scripts/plot_calibration.py --input data/test_predictions/test_predictions.parquet
    python scripts/plot_calibration.py --num-bins 30 --no-wandb
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import wandb

from intention_jailbreak.utils.calibration import (
    classifier_calibration_curve,
    classifier_calibration_error,
    plot_calibration_curve
)


def main():
    parser = argparse.ArgumentParser(description="Plot calibration curve from saved predictions")
    parser.add_argument(
        "--input",
        type=str,
        default="data/test_predictions/test_predictions.parquet",
        help="Path to predictions file (parquet format)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for calibration plot (default: same directory as input)"
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=20,
        help="Number of bins for calibration curve"
    )
    parser.add_argument(
        "--prob-column",
        type=str,
        default="harmful_probability",
        help="Name of the probability column"
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="label",
        help="Name of the true label column"
    )
    parser.add_argument(
        "--pred-column",
        type=str,
        default="predicted_label",
        help="Name of the predicted label column"
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Skip wandb logging"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="classifier-annotation-set",
        help="WandB project name"
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default="intention-analysis",
        help="WandB entity name"
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Classifier Calibration Curve",
        help="Plot title"
    )
    
    args = parser.parse_args()
    
    # Load predictions
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    print(f"Loading predictions from: {input_path}")
    df = pd.read_parquet(input_path)
    
    # Verify required columns exist
    required_columns = [args.prob_column, args.label_column, args.pred_column]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    print(f"Loaded {len(df)} predictions")
    
    # Extract data
    y_pred = df[args.pred_column].values
    y_true = df[args.label_column].values
    y_probs = df[args.prob_column].values
    
    # Initialize wandb if requested
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name="calibration-plot",
            dir="logs/wandb",
            config={
                "input_file": str(input_path),
                "num_bins": args.num_bins,
                "num_samples": len(df)
            }
        )
    
    # Compute calibration curve
    print(f"\nComputing calibration curve with {args.num_bins} bins...")
    conf_curve, acc_curve = classifier_calibration_curve(
        y_pred=y_pred,
        y_true=y_true,
        y_confidences=y_probs,
        num_bins=args.num_bins
    )
    
    # Compute calibration error
    cal_error = classifier_calibration_error(
        y_pred=y_pred,
        y_true=y_true,
        y_confidences=y_probs,
        num_bins=args.num_bins,
    )
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / "classifier_calibration_curve.png"
    
    # Plot calibration curve
    print(f"Generating calibration plot...")
    plot_calibration_curve(
        conf=conf_curve,
        acc=acc_curve,
        title=args.title,
        y_label="Empirical Accuracy",
        relative_save_path=output_path
    )
    
    print(f"\nCalibration Error (MAE): {cal_error:.4f}")
    print(f"Calibration plot saved to: {output_path}")
    
    # Log metrics to wandb
    if not args.no_wandb:
        wandb.log({
            "calibration_error": cal_error,
        })
        
        # Log artifact
        artifact = wandb.Artifact(
            name="calibration-curve",
            type="figure",
            description=f"Calibration curve with {args.num_bins} bins"
        )
        artifact.add_file(output_path)
        wandb.log_artifact(artifact)
        
        wandb.finish()
    
    # Print calibration analysis
    print("\n" + "="*60)
    print("CALIBRATION ANALYSIS")
    print("="*60)
    
    # Check if model is over/under confident
    mean_diff = np.mean(np.array(acc_curve) - np.array(conf_curve))
    if mean_diff > 0.05:
        print("⚠️  Model is UNDERCONFIDENT")
        print("   → Predicted probabilities are lower than actual accuracy")
        print("   → Consider applying inverse temperature scaling")
    elif mean_diff < -0.05:
        print("⚠️  Model is OVERCONFIDENT")
        print("   → Predicted probabilities are higher than actual accuracy")
        print("   → Consider applying temperature scaling or Platt scaling")
    else:
        print("✓ Model is reasonably well-calibrated")
    
    print(f"\nMean calibration difference: {mean_diff:+.4f}")
    print(f"Calibration error (MAE): {cal_error:.4f}")
    
    # Calibration quality assessment
    if cal_error < 0.1:
        quality = "Excellent"
    elif cal_error < 0.15:
        quality = "Good"
    elif cal_error < 0.2:
        quality = "Moderate"
    else:
        quality = "Poor"
    
    print(f"Overall calibration quality: {quality}")
    print("="*60)


if __name__ == "__main__":
    main()
