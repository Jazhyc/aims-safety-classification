# Ablations

This directory holds HPC orchestration for ablation experiments that probe
specific data-construction choices in the main distillation pipeline. Each
ablation is its own self-contained config under
`configs/experiments/ablations/` plus a `submit_*.py` orchestrator here.

## label_source_ablation

**Question:** how much of the main result is driven by (a) human relabeling of
the harm class and (b) selecting the *hard* uncertainty-filtered subset?

**Design.** Three conditions on the same synthetic_intent distillation
pipeline (gpt-oss-120b → gemma-3-12b, lr=2e-5, seed=42):

| Condition | Train prompts | Train labels | Source |
|---|---|---|---|
| `hard_human` | annotated-intents `train` (all rows) | 5-level human "Annotator Harm" | **existing main result, not re-run** |
| `hard_original` | annotated-intents `train` deduped (one row per Prompt) | binary WG `Dataset Harm` (mapped: Harmful→harmful, Safe→unharmful) | this orchestrator |
| `random_original` | random sample of `allenai/wildguardmix` (wildguardtrain), excludes annotated prompts, stratified on (prompt_harm_label, adversarial), size-matched to hard_original | original WG `prompt_harm_label` | this orchestrator |

`hard_original` vs `hard_human` isolates the relabeling effect; `random_original`
vs `hard_original` isolates the hard-subset effect.

**Why one seed and one LR.** Both conditions share the per-combo best LR from
the v7 sweep (`2e-5`, the W&B-best for gpt-oss-120b → gemma-3-12b →
synthetic_intent). Variance across seeds is expected to be small relative to
the data-source effect being measured.

**Why no teacher-disagreement filter.** On the hard subset the gpt-oss-120b
teacher matches the human harm label on ~5/1300 examples, so the
hard_human-vs-hard_original difference is dominated by the teacher's reasoning
*text* (which is conditioned on the supplied label via the
`teacher_ground_truth` template), not by which examples get filtered. Keeping
the filter off matches the existing v7 sweep.

**Held-out signal during training.** The v7 validation/test traces under
`data/reasoning_traces_v7/openai-gpt-oss-120b/{validation,test}/parsed_results.json`
are reused as the early-stopping signal for both new conditions. Held-out
splits are not regenerated because val/test in distillation is just a loss/F1
monitor and teacher-vs-GT disagreement on those splits is small.

### Pipeline

```bash
cd /scratch/s4626451/intention-jailbreak

# 1. Build the two train sample JSONs (~30s, runs locally)
python scripts/hpc/ablations/submit_label_source_ablation.py --mode samples

# 2. Generate gpt-oss-120b reasoning traces for the two new train sets
python scripts/hpc/ablations/submit_label_source_ablation.py --mode traces

# 3. After traces finish, train the gemma-3-12b student on each condition
python scripts/hpc/ablations/submit_label_source_ablation.py --mode train

# 4. After training, run OOD validation (ToxicChat + Aegis)
python scripts/hpc/ablations/submit_label_source_ablation.py --mode ood-val

# 5. Then full test eval
python scripts/hpc/ablations/submit_label_source_ablation.py --mode test
```

Every mode supports `--dry-run` (preview) and `--force` (re-submit even when
outputs already exist).

### Outputs

- Sample JSONs:  `data/reasoning_traces_v7_ablations/samples/`
- Teacher traces: `data/reasoning_traces_v7_ablations/openai-gpt-oss-120b/{hard_original,random_original}/train/`
- Adapters: `models/distillation-ablations/openai-gpt-oss-120b--gemma-3-12b--synthetic-intent--{hard-original,random-original}--lr2e-05--v7-ablation_adapter/`
- OOD val:  `data/safety_experiment/ood_validation/distillation-ablations/{hard_original,random_original}/{adapter_name}/{toxic-chat,aegis}/`
- Test eval: `data/safety_experiment/distillation-ablations/{hard_original,random_original}/`

The adapter directories are intentionally placed outside
`models/distillation-sweep/` so the existing distillation eval submitter
doesn't pick them up.
