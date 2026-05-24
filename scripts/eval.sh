#!/bin/bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════
# ConGra 평가 스크립트
# GPU 인스턴스에서 실행
#
# 단일 모드:           bash scripts/eval.sh --mode baseline
# Qwen 1.5B:           bash scripts/eval.sh --ablation --qwen-1.5b
# 전체 ablation:       bash scripts/eval.sh --ablation
# zero-shot만:         bash scripts/eval.sh --zeroshot
# ablation + zeroshot: bash scripts/eval.sh --ablation --zeroshot
# ═══════════════════════════════════════════════════════════════

# ── 실험 설정 ─────────────────────────────────────────────────
MODE="baseline"
CONTEXT_LINES=20
SPLIT="test"
MAX_NEW_TOKENS=512
MAX_SAMPLES=""           # 빈 문자열이면 전체 평가
RUN_BASELINE_COMPARE=false
ABLATION=false
ZEROSHOT=false

BASE_MODEL_NAME="deepseek-ai/deepseek-coder-1.3b-base"
MODEL_TAG="deepseek-coder-1.3b-base"
CKPT_NAME_SUFFIX=""

# ── 인자 파싱 ─────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)        MODE="$2";             shift 2 ;;
        --ctx)         CONTEXT_LINES="$2";    shift 2 ;;
        --split)       SPLIT="$2";            shift 2 ;;
        --max-samples) MAX_SAMPLES="$2";      shift 2 ;;
        --compare)     RUN_BASELINE_COMPARE=true; shift ;;
        --ablation)    ABLATION=true;         shift ;;
        --zeroshot)    ZEROSHOT=true;         shift ;;
        --base-model)  BASE_MODEL_NAME="$2"; MODEL_TAG="$(basename "$2" | tr '[:upper:]' '[:lower:]')"; shift 2 ;;
        --qwen-1.5b)   BASE_MODEL_NAME="/opt/dlami/nvme/models/qwen2.5-coder-1.5b"; MODEL_TAG="qwen2.5-coder-1.5b"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ "${MODEL_TAG}" != "deepseek-coder-1.3b-base" ]; then
    CKPT_NAME_SUFFIX="_${MODEL_TAG}"
fi
RESULTS_ROOT="./eval_results/${MODEL_TAG}"

# ── Python 환경 선택 ──────────────────────────────────────────
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! ${PYTHON_BIN} -c "import torch" >/dev/null 2>&1; then
    if command -v conda >/dev/null 2>&1 && conda run -n final_project python -c "import torch" >/dev/null 2>&1; then
        PYTHON_BIN="conda run --no-capture-output -n final_project python"
        echo "[INFO] Using conda env: final_project"
    else
        echo "[ERROR] torch not found in ${PYTHON_BIN}. Activate/install the eval env first."
        echo "        Try: conda activate final_project"
        exit 1
    fi
fi

# ── 로그 ──────────────────────────────────────────────────────
LOG_DIR="./logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ── NVMe 모델 복구/다운로드 ──────────────────────────────────
ensure_model_available() {
    local model_path="$1"
    if [[ "${model_path}" != /* ]]; then
        return 0
    fi
    if [ -f "${model_path}/config.json" ]; then
        return 0
    fi

    case "${MODEL_TAG}" in
        qwen2.5-coder-1.5b)
            echo ""
            echo "[Model] Local model missing: ${model_path}"
            echo "[Model] Restoring/downloading Qwen 1.5B to NVMe"
            bash scripts/download_model.sh --qwen-1.5b
            ;;
        deepseek-coder-6.7b-base)
            echo ""
            echo "[Model] Local model missing: ${model_path}"
            echo "[Model] Restoring/downloading DeepSeek Coder 6.7B to NVMe"
            bash scripts/download_model.sh --6.7b
            ;;
        deepseek-coder-1.3b-base)
            echo ""
            echo "[Model] Local model missing: ${model_path}"
            echo "[Model] Restoring/downloading DeepSeek Coder 1.3B to NVMe"
            bash scripts/download_model.sh --1.3b
            ;;
        *)
            echo "[ERROR] Local model path missing and no restore preset is known: ${model_path}"
            exit 1
            ;;
    esac

    if [ ! -f "${model_path}/config.json" ]; then
        echo "[ERROR] Model restore failed: ${model_path}"
        exit 1
    fi
}

_detect_base_model() {
    for m in baseline type ast "ast+type"; do
        local cfg="./ckpts/${m}_ctx${CONTEXT_LINES}${CKPT_NAME_SUFFIX}/final/adapter_config.json"
        if [ -f "${cfg}" ]; then
            python3 -c "import json; print(json.load(open('${cfg}'))['base_model_name_or_path'])"
            return
        fi
    done
    echo "${BASE_MODEL_NAME}"
}

run_zeroshot() {
    local m="$1"
    local DATASET_DIR="./data/processed/dataset_${m}_ctx${CONTEXT_LINES}"
    local EVAL_OUTPUT_DIR="${RESULTS_ROOT}/${m}_ctx${CONTEXT_LINES}"

    if [ ! -d "${DATASET_DIR}" ]; then
        echo "[ERROR] Dataset not found: ${DATASET_DIR}"
        return 1
    fi

    mkdir -p "${EVAL_OUTPUT_DIR}"

    local base_model
    base_model=$(_detect_base_model)
    ensure_model_available "${base_model}"

    echo "========================================"
    echo " Zero-shot Evaluation"
    echo "========================================"
    echo " Mode:        ${m}"
    echo " Model tag:   ${MODEL_TAG}"
    echo " Base model:  ${base_model}"
    echo " Dataset:     ${DATASET_DIR}"
    echo " Split:       ${SPLIT}"
    echo " Output:      ${EVAL_OUTPUT_DIR}"
    echo "========================================"
    echo ""

    CMD="${PYTHON_BIN} evaluate.py \
        --model_name ${base_model} \
        --dataset_dir ${DATASET_DIR} \
        --split ${SPLIT} \
        --max_new_tokens ${MAX_NEW_TOKENS} \
        --output_dir ${EVAL_OUTPUT_DIR}"

    [ -n "${MAX_SAMPLES}" ] && CMD="${CMD} --max_samples ${MAX_SAMPLES}"
    eval ${CMD}

    echo ""
    echo "Zero-shot results saved to: ${EVAL_OUTPUT_DIR}/metrics_base.json"
}

run_eval() {
    local m="$1"
    local DATASET_DIR="./data/processed/dataset_${m}_ctx${CONTEXT_LINES}"
    local MODEL_DIR="./ckpts/${m}_ctx${CONTEXT_LINES}${CKPT_NAME_SUFFIX}/final"
    local EVAL_OUTPUT_DIR="${RESULTS_ROOT}/${m}_ctx${CONTEXT_LINES}"

    if [ ! -d "${DATASET_DIR}" ]; then
        echo "[ERROR] Dataset not found: ${DATASET_DIR}"
        return 1
    fi
    if [ ! -d "${MODEL_DIR}" ]; then
        echo "[ERROR] Model not found: ${MODEL_DIR}"
        echo "  Run train.sh --mode ${m} first."
        return 1
    fi

    mkdir -p "${EVAL_OUTPUT_DIR}"
    ensure_model_available "${BASE_MODEL_NAME}"

    echo "========================================"
    echo " ConGra Evaluation"
    echo "========================================"
    echo " Mode:        ${m}"
    echo " Model tag:   ${MODEL_TAG}"
    echo " Model:       ${MODEL_DIR}"
    echo " Base model:  ${BASE_MODEL_NAME}"
    echo " Dataset:     ${DATASET_DIR}"
    echo " Split:       ${SPLIT}"
    echo " Max tokens:  ${MAX_NEW_TOKENS}"
    echo " Output:      ${EVAL_OUTPUT_DIR}"
    echo "========================================"
    echo ""

    CMD="${PYTHON_BIN} evaluate.py \
        --model_dir ${MODEL_DIR} \
        --base_model_name ${BASE_MODEL_NAME} \
        --dataset_dir ${DATASET_DIR} \
        --split ${SPLIT} \
        --max_new_tokens ${MAX_NEW_TOKENS} \
        --output_dir ${EVAL_OUTPUT_DIR}"

    [ -n "${MAX_SAMPLES}" ] && CMD="${CMD} --max_samples ${MAX_SAMPLES}"
    eval ${CMD}

    if [ "${RUN_BASELINE_COMPARE}" = true ]; then
        echo ""
        run_zeroshot "${m}"

        echo ""
        echo "========================================"
        echo " Comparison Summary (${m})"
        echo "========================================"
        ${PYTHON_BIN} -c "
import json
with open('${EVAL_OUTPUT_DIR}/metrics_finetuned.json') as f:
    ft = json.load(f)['overall']
with open('${EVAL_OUTPUT_DIR}/metrics_base.json') as f:
    base = json.load(f)['overall']
print(f\"{'Metric':<20} {'Zero-shot':>10} {'Fine-tuned':>10} {'Delta':>10}\")
print('-' * 52)
for key in ['exact_match_rate', 'avg_bleu', 'avg_codebleu', 'avg_token_f1', 'avg_chrf', 'avg_edit_distance']:
    if key not in base or key not in ft:
        continue
    b, f_ = base[key], ft[key]
    d = f_ - b
    sign = '+' if d > 0 else ''
    better = '↑' if (d > 0 and key != 'avg_edit_distance') or (d < 0 and key == 'avg_edit_distance') else '↓' if d != 0 else ''
    print(f'{key:<22} {b:>10.4f} {f_:>10.4f} {sign}{d:>9.4f} {better}')
"
    fi

    echo ""
    echo "Results saved to: ${EVAL_OUTPUT_DIR}/"
}

if [ "${ABLATION}" = true ]; then
    MODES=("baseline" "type" "ast" "ast+type")
    echo "========================================"
    if [ "${ZEROSHOT}" = true ]; then
        echo " [Ablation] 전체 모드 순차 평가 (zero-shot 포함)"
    else
        echo " [Ablation] 전체 모드 순차 평가"
    fi
    echo " Modes: ${MODES[*]}"
    echo " ctx:   ${CONTEXT_LINES}"
    echo " model: ${MODEL_TAG}"
    echo " out:   ${RESULTS_ROOT}"
    echo "========================================"
else
    MODES=("${MODE}")
fi

for m in "${MODES[@]}"; do
    FINETUNED_RESULT="${RESULTS_ROOT}/${m}_ctx${CONTEXT_LINES}/metrics_finetuned.json"
    BASE_RESULT="${RESULTS_ROOT}/${m}_ctx${CONTEXT_LINES}/metrics_base.json"

    if [ "${ZEROSHOT}" = true ] && [ "${ABLATION}" = false ]; then
        LOG_FILE="${LOG_DIR}/eval_zeroshot_${MODEL_TAG}_${m}_ctx${CONTEXT_LINES}_${SPLIT}_${TIMESTAMP}.log"
        echo "[Log] ${LOG_FILE}"
        (
            exec > >(tee -a "${LOG_FILE}") 2>&1
            run_zeroshot "${m}"
        )
    elif [ "${ZEROSHOT}" = true ]; then
        if [ "${ABLATION}" = true ] && [ -f "${BASE_RESULT}" ]; then
            echo "[Skip] ${m}: zero-shot 결과 이미 존재 → ${BASE_RESULT}"
        else
            LOG_FILE="${LOG_DIR}/eval_zeroshot_${MODEL_TAG}_${m}_ctx${CONTEXT_LINES}_${SPLIT}_${TIMESTAMP}.log"
            echo "[Log] ${LOG_FILE}"
            (
                exec > >(tee -a "${LOG_FILE}") 2>&1
                run_zeroshot "${m}"
            )
        fi
        if [ "${ABLATION}" = true ] && [ -f "${FINETUNED_RESULT}" ]; then
            echo "[Skip] ${m}: 파인튜닝 결과 이미 존재 → ${FINETUNED_RESULT}"
        else
            LOG_FILE="${LOG_DIR}/eval_${MODEL_TAG}_${m}_ctx${CONTEXT_LINES}_${SPLIT}_${TIMESTAMP}.log"
            echo "[Log] ${LOG_FILE}"
            (
                exec > >(tee -a "${LOG_FILE}") 2>&1
                run_eval "${m}"
            )
        fi
    else
        if [ "${ABLATION}" = true ] && [ -f "${FINETUNED_RESULT}" ]; then
            echo "[Skip] ${m}: 파인튜닝 결과 이미 존재 → ${FINETUNED_RESULT}"
        else
            LOG_FILE="${LOG_DIR}/eval_${MODEL_TAG}_${m}_ctx${CONTEXT_LINES}_${SPLIT}_${TIMESTAMP}.log"
            echo "[Log] ${LOG_FILE}"
            (
                exec > >(tee -a "${LOG_FILE}") 2>&1
                run_eval "${m}"
            )
        fi
    fi
done

if [ "${ABLATION}" = true ]; then
    echo ""
    echo "========================================"
    echo " [Ablation] 결과 요약"
    echo "========================================"
    ${PYTHON_BIN} -c "
import json, os
modes = ['baseline', 'type', 'ast', 'ast+type']
ctx = ${CONTEXT_LINES}
root = '${RESULTS_ROOT}'
has_zeroshot = ${ZEROSHOT^}
keys = ['exact_match_rate', 'avg_bleu', 'avg_codebleu', 'avg_token_f1', 'avg_chrf', 'avg_edit_distance']
labels = ['EM', 'BLEU', 'CodeBLEU', 'Token-F1', 'chrF', 'EditDist']
print(f\"{'Model':<18}\" + ''.join(f'{l:>12}' for l in labels))
print('-' * (18 + 12 * len(keys)))
if has_zeroshot:
    for m in modes:
        path = f'{root}/{m}_ctx{ctx}/metrics_base.json'
        if not os.path.exists(path):
            continue
        with open(path) as f:
            overall = json.load(f)['overall']
        print(f'zero-shot/{m:<8}' + ''.join(f'{overall[k]:>12.4f}' for k in keys))
    print('-' * (18 + 12 * len(keys)))
for m in modes:
    path = f'{root}/{m}_ctx{ctx}/metrics_finetuned.json'
    if not os.path.exists(path):
        print(f'{m:<18}  (결과 없음)')
        continue
    with open(path) as f:
        overall = json.load(f)['overall']
    print(f'{m:<18}' + ''.join(f'{overall[k]:>12.4f}' for k in keys))
"
    echo "========================================"
fi
