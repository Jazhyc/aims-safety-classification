#!/bin/bash
# Quick pre-flight check for DPO training environment.
# Run on a LOGIN node before submitting jobs:
#   bash scripts/hpc/check_dpo_env.sh
#
# All checks print OK or FAIL. Fix any FAILs before submitting.

set -uo pipefail

PROJECTS_DIR="${PROJECTS_DIR:-/scratch/s4351495}"
HF_HOME_CHECK="${HF_HOME:-${PROJECTS_DIR}/huggingface_cache}"
SFT_ADAPTER="${SFT_ADAPTER:-${PROJECTS_DIR}/trained_models/causal/hyperparam_sweep/lr_0.0005_e_5_adapter}"
BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
MODEL_ID="${BASE_MODEL//\//-}"   # meta-llama--Llama-3.1-8B-Instruct

pass() { echo "[OK]   $1"; }
fail() { echo "[FAIL] $1"; FAILED=1; }
FAILED=0

echo "========================================"
echo "DPO environment pre-flight check"
echo "PROJECTS_DIR : $PROJECTS_DIR"
echo "HF_HOME      : $HF_HOME_CHECK"
echo "SFT_ADAPTER  : $SFT_ADAPTER"
echo "========================================"

# 1. HF cache directory
SNAP_DIR="$HF_HOME_CHECK/hub/models--${MODEL_ID}/snapshots"
if [ -d "$SNAP_DIR" ]; then
    SNAP=$(ls "$SNAP_DIR" | tail -1)
    pass "HF snapshot exists: $SNAP_DIR/$SNAP"
    # Check key files
    for f in config.json tokenizer.json tokenizer_config.json; do
        if [ -f "$SNAP_DIR/$SNAP/$f" ]; then
            pass "  $f present"
        else
            fail "  $f MISSING from snapshot"
        fi
    done
else
    fail "HF snapshot dir not found: $SNAP_DIR"
fi

# 2. SFT adapter
if [ -d "$SFT_ADAPTER" ]; then
    pass "SFT adapter dir exists"
    if [ -f "$SFT_ADAPTER/adapter_config.json" ]; then
        pass "  adapter_config.json present"
    else
        fail "  adapter_config.json MISSING"
    fi
else
    fail "SFT adapter not found: $SFT_ADAPTER"
fi

# 3. Balanced DPO pairs
PAIRS="data/dpo_pairs/train_t0.8_balanced/dpo_pairs.jsonl"
if [ -f "$PAIRS" ]; then
    N=$(wc -l < "$PAIRS")
    pass "Balanced pairs file exists ($N lines)"
else
    fail "Balanced pairs not found: $PAIRS"
fi

# 4. Python tokenizer load test
echo ""
echo "--- Tokenizer load test ---"
source .venv/bin/activate 2>/dev/null || true
export HF_HOME="$HF_HOME_CHECK"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python3 - <<'PYEOF'
import os, sys
hf_home = os.environ["HF_HOME"]
model_id = "meta-llama--Llama-3.1-8B-Instruct"
snapshots = os.path.join(hf_home, "hub", f"models--{model_id}", "snapshots")
snap = sorted(os.listdir(snapshots))[-1]
tok_path = os.path.join(snapshots, snap)
print(f"  Loading tokenizer from: {tok_path}")
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(tok_path, local_files_only=True)
print(f"  Tokenizer class : {tok.__class__.__name__}")
print(f"  Vocab size      : {tok.vocab_size}")
print("[OK]   Tokenizer loads successfully")
PYEOF

if [ $? -ne 0 ]; then
    FAILED=1
    echo "[FAIL] Tokenizer load test failed"
fi

echo ""
echo "========================================"
if [ "$FAILED" -eq 0 ]; then
    echo "All checks passed — safe to sbatch"
else
    echo "Some checks FAILED — fix before submitting"
fi
echo "========================================"
