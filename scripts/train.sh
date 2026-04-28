#!/bin/bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════
# ConGra QLoRA 학습 스크립트
# GPU 인스턴스(L4 24GB)에서 실행
# ═══════════════════════════════════════════════════════════════

# ── 실험 설정 ─────────────────────────────────────────────────
# 전처리 모드에 맞춰 dataset_dir을 바꿔주세요
MODE="baseline"
CONTEXT_LINES=10
DATASET_DIR="./data/processed/dataset_${MODE}_ctx${CONTEXT_LINES}"

# 모델
MODEL_NAME="deepseek-ai/deepseek-coder-1.3b-base"
OUTPUT_DIR="./output/${MODE}_ctx${CONTEXT_LINES}"

# 학습 하이퍼파라미터
BATCH_SIZE=8
GRAD_ACCUM=2             # effective batch = BATCH_SIZE * GRAD_ACCUM
NUM_EPOCHS=10            # early stopping이 실제 제어
LR=2e-4
MAX_SEQ_LEN=2048

# LoRA
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05

# Early stopping / eval
EVAL_STEPS=200
SAVE_STEPS=200
PATIENCE=3

# Misc
SEED=42
USE_WANDB=false

# ── 인자 파싱 (선택) ─────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)       MODE="$2";          DATASET_DIR="./data/processed/dataset_${MODE}_ctx${CONTEXT_LINES}"; OUTPUT_DIR="./output/${MODE}_ctx${CONTEXT_LINES}"; shift 2 ;;
        --ctx)        CONTEXT_LINES="$2"; DATASET_DIR="./data/processed/dataset_${MODE}_ctx${CONTEXT_LINES}"; OUTPUT_DIR="./output/${MODE}_ctx${CONTEXT_LINES}"; shift 2 ;;
        --model)      MODEL_NAME="$2";    shift 2 ;;
        --batch)      BATCH_SIZE="$2";    shift 2 ;;
        --lr)         LR="$2";           shift 2 ;;
        --wandb)      USE_WANDB=true;     shift   ;;
        --resume)     RESUME="$2";        shift 2 ;;
        # 6.7B 프리셋
        --6.7b)       MODEL_NAME="deepseek-ai/deepseek-coder-6.7b-base"; BATCH_SIZE=2; GRAD_ACCUM=8; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── 사전 체크 ─────────────────────────────────────────────────
if [ ! -d "${DATASET_DIR}" ]; then
    echo "[ERROR] Dataset not found: ${DATASET_DIR}"
    echo "  Run preprocess.sh --mode ${MODE} --ctx ${CONTEXT_LINES} first."
    exit 1
fi

# ── 요약 출력 ─────────────────────────────────────────────────
echo "========================================"
echo " ConGra QLoRA Training"
echo "========================================"
echo " Model:         ${MODEL_NAME}"
echo " Dataset:       ${DATASET_DIR}"
echo " Output:        ${OUTPUT_DIR}"
echo " Batch:         ${BATCH_SIZE} x ${GRAD_ACCUM} = $((BATCH_SIZE * GRAD_ACCUM))"
echo " LR:            ${LR}"
echo " Epochs:        ${NUM_EPOCHS} (patience=${PATIENCE})"
echo " Max seq len:   ${MAX_SEQ_LEN}"
echo " LoRA:          r=${LORA_R}, alpha=${LORA_ALPHA}"
echo " WandB:         ${USE_WANDB}"
echo "========================================"
echo ""

# ── 실행 ──────────────────────────────────────────────────────
CMD="python3 train.py \
    --dataset_dir ${DATASET_DIR} \
    --model_name ${MODEL_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --num_epochs ${NUM_EPOCHS} \
    --learning_rate ${LR} \
    --max_seq_length ${MAX_SEQ_LEN} \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout ${LORA_DROPOUT} \
    --eval_steps ${EVAL_STEPS} \
    --save_steps ${SAVE_STEPS} \
    --early_stopping_patience ${PATIENCE} \
    --seed ${SEED}"

if [ "${USE_WANDB}" = true ]; then
    CMD="${CMD} --use_wandb --wandb_project congra-merge-conflict"
fi

if [ -n "${RESUME:-}" ]; then
    CMD="${CMD} --resume_from_checkpoint ${RESUME}"
fi

eval ${CMD}
