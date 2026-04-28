#!/bin/bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════
# ConGra 평가 스크립트
# GPU 인스턴스에서 실행
# ═══════════════════════════════════════════════════════════════

# ── 실험 설정 ─────────────────────────────────────────────────
MODE="baseline"
CONTEXT_LINES=10
DATASET_DIR="./data/processed/dataset_${MODE}_ctx${CONTEXT_LINES}"
MODEL_DIR="./output/${MODE}_ctx${CONTEXT_LINES}/final"
EVAL_OUTPUT_DIR="./eval_results/${MODE}_ctx${CONTEXT_LINES}"

SPLIT="test"
MAX_NEW_TOKENS=512
MAX_SAMPLES=""           # 빈 문자열이면 전체 평가

# ── 인자 파싱 ─────────────────────────────────────────────────
RUN_BASELINE_COMPARE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)       MODE="$2"; shift 2
                      DATASET_DIR="./data/processed/dataset_${MODE}_ctx${CONTEXT_LINES}"
                      MODEL_DIR="./output/${MODE}_ctx${CONTEXT_LINES}/final"
                      EVAL_OUTPUT_DIR="./eval_results/${MODE}_ctx${CONTEXT_LINES}" ;;
        --ctx)        CONTEXT_LINES="$2"; shift 2
                      DATASET_DIR="./data/processed/dataset_${MODE}_ctx${CONTEXT_LINES}"
                      MODEL_DIR="./output/${MODE}_ctx${CONTEXT_LINES}/final"
                      EVAL_OUTPUT_DIR="./eval_results/${MODE}_ctx${CONTEXT_LINES}" ;;
        --model-dir)  MODEL_DIR="$2";       shift 2 ;;
        --split)      SPLIT="$2";           shift 2 ;;
        --max-samples) MAX_SAMPLES="$2";    shift 2 ;;
        --compare)    RUN_BASELINE_COMPARE=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── 사전 체크 ─────────────────────────────────────────────────
if [ ! -d "${DATASET_DIR}" ]; then
    echo "[ERROR] Dataset not found: ${DATASET_DIR}"
    exit 1
fi
if [ ! -d "${MODEL_DIR}" ]; then
    echo "[ERROR] Model not found: ${MODEL_DIR}"
    echo "  Run train.sh --mode ${MODE} first."
    exit 1
fi

mkdir -p "${EVAL_OUTPUT_DIR}"

# ── 요약 출력 ─────────────────────────────────────────────────
echo "========================================"
echo " ConGra Evaluation"
echo "========================================"
echo " Model:       ${MODEL_DIR}"
echo " Dataset:     ${DATASET_DIR}"
echo " Split:       ${SPLIT}"
echo " Max tokens:  ${MAX_NEW_TOKENS}"
echo " Output:      ${EVAL_OUTPUT_DIR}"
echo "========================================"
echo ""

# ── 1) 파인튜닝 모델 평가 ────────────────────────────────────
echo ">>> Evaluating fine-tuned model..."

CMD="python3 evaluate.py \
    --model_dir ${MODEL_DIR} \
    --dataset_dir ${DATASET_DIR} \
    --split ${SPLIT} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --output_dir ${EVAL_OUTPUT_DIR}"

if [ -n "${MAX_SAMPLES}" ]; then
    CMD="${CMD} --max_samples ${MAX_SAMPLES}"
fi

eval ${CMD}

# ── 2) 베이스 모델 비교 (--compare 옵션) ─────────────────────
if [ "${RUN_BASELINE_COMPARE}" = true ]; then
    echo ""
    echo ">>> Evaluating base model (for comparison)..."

    # adapter_config.json에서 base model name 추출
    BASE_MODEL=$(python3 -c "
import json
with open('${MODEL_DIR}/adapter_config.json') as f:
    print(json.load(f)['base_model_name_or_path'])
")

    CMD_BASE="python evaluate.py \
        --model_name ${BASE_MODEL} \
        --dataset_dir ${DATASET_DIR} \
        --split ${SPLIT} \
        --max_new_tokens ${MAX_NEW_TOKENS} \
        --output_dir ${EVAL_OUTPUT_DIR}"

    if [ -n "${MAX_SAMPLES}" ]; then
        CMD_BASE="${CMD_BASE} --max_samples ${MAX_SAMPLES}"
    fi

    eval ${CMD_BASE}

    # ── 비교 요약 ─────────────────────────────────────────────
    echo ""
    echo "========================================"
    echo " Comparison Summary"
    echo "========================================"
    python3 -c "
import json

with open('${EVAL_OUTPUT_DIR}/metrics_finetuned.json') as f:
    ft = json.load(f)['overall']
with open('${EVAL_OUTPUT_DIR}/metrics_base.json') as f:
    base = json.load(f)['overall']

print(f\"{'Metric':<20} {'Base':>10} {'Fine-tuned':>10} {'Delta':>10}\")
print('-' * 52)
for key in ['exact_match_rate', 'avg_bleu', 'avg_edit_distance']:
    b, f_ = base[key], ft[key]
    d = f_ - b
    sign = '+' if d > 0 else ''
    better = '↑' if (d > 0 and key != 'avg_edit_distance') or (d < 0 and key == 'avg_edit_distance') else '↓' if d != 0 else ''
    print(f'{key:<20} {b:>10.4f} {f_:>10.4f} {sign}{d:>9.4f} {better}')
"
fi

echo ""
echo "Results saved to: ${EVAL_OUTPUT_DIR}/"
