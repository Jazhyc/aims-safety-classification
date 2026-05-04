# Data-scaling experiment

Generates gpt-oss-120b reasoning traces for the entire WildGuardMix
`wildguardtrain` split.

Annotated-intents prompts are NOT excluded from the pool: that dataset was
originally a training/validation source rather than a primary held-out test,
and the held-out mean reported in the eval notebook already excludes it.
The 5 genuinely OOD eval datasets (ToxicChat, Aegis, XSTest, OpenAI
Moderation, WildGuardMix-test) remain valid for scaling-vs-baseline
comparison; only the `annotated-intents` test column becomes
in-distribution and should not be cross-compared with adapters that didn't
see those prompts during training.

Motivated by the label-source ablation: at n=932 a random WG sample beat the
human-curated hard subset on held-out test F1, suggesting the gemma-3-12b
student is undertrained rather than starved of high-quality data. This
experiment builds the full trace corpus so we can later distill at multiple
sample sizes and trace out the data-scaling curve.

## Pipeline

```bash
cd /scratch/s4626451/intention-jailbreak

# 1. Build the full WG sample JSON (~86.7k records, runs locally)
python scripts/hpc/scaling/submit_scaling.py --mode samples

# 2. Submit the SLURM trace-gen job (24h budget, ~6h expected at observed throughput)
python scripts/hpc/scaling/submit_scaling.py --mode traces

# 3. After traces finish, submit the SLURM training job
#    (24h budget, 1 epoch ≈ 8h on rtx_pro_6000, no early stopping needed at this scale)
python scripts/hpc/scaling/submit_scaling.py --mode train

# 4. After training, submit the test eval (all 6 EVAL_DATASET_DIRS)
python scripts/hpc/scaling/submit_scaling.py --mode test
```

All modes accept `--dry-run` (preview) and `--force` (re-run even when
outputs exist).

### Training notes

- **One epoch** at 86k records is already 92× the data the 5-epoch ablation
  runs saw. Early stopping is left enabled by the underlying training script
  but is inert with `epochs=1` (a single eval cannot trigger it).
- **No HP sweep**, no model selection — same per-combo best LR (2e-5) as the
  label-source ablation.
- **Sequence packing is NOT enabled** in this template. Speed-test packing
  on an interactive node first; if cleanly supported by the trainer's
  flash-attn varlen path, flipping it on can give roughly a 6× wall-clock
  speedup at the cost of needing to re-validate the loss curve matches the
  unpacked baseline.


## Outputs

- Sample JSON: `data/reasoning_traces_scaling/samples/full_wg_train_samples.json`
- Traces: `data/reasoning_traces_scaling/openai-gpt-oss-120b/full_wg/train/{raw_outputs,parsed_results}.json`

The output base dir is intentionally distinct from `data/reasoning_traces_v7/`
and `data/reasoning_traces_v7_ablations/` so the existing distillation eval
submitter (which discovers adapters by trace-version naming) doesn't pick this
up.

## Notes

- Single (teacher, condition): `gpt-oss-120b` × `synthetic_intent`.
- Sample order is shuffled with `seed=42` (only affects record order in the
  JSON, not which prompts are included).
