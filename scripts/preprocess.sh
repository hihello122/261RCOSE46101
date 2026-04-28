#!/bin/bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════
# ConGra 전처리 스크립트
# --mode all 로 4개 ablation 한번에 생성 가능
# ═══════════════════════════════════════════════════════════════

# ── 설정 ──────────────────────────────────────────────────────
DATA_DIR="./data/raw_datasets/Java"
OUTPUT_DIR="./data/processed"
CONTEXT_LINES=20
MODE="baseline"          # baseline | ast | type | ast+type | all
MAX_SEQ_LEN=2048
TOKENIZER="deepseek-ai/deepseek-coder-1.3b-base"
GUMTREE_BIN="./gumtree-3.0.0/bin/gumtree"   # ast 모드에서만 사용
SKIP_STATS=false

# ── 인자 파싱 ─────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)       MODE="$2";          shift 2 ;;
        --ctx)        CONTEXT_LINES="$2"; shift 2 ;;
        --skip-stats) SKIP_STATS=true;    shift   ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── 모드 목록 결정 ────────────────────────────────────────────
if [ "${MODE}" = "all" ]; then
    MODES=("baseline" "type")
    # ast 모드는 gumtree 설치되어 있을 때만
    if command -v "${GUMTREE_BIN}" &> /dev/null; then
        MODES+=("ast" "ast+type")
    else
        echo "[WARN] GumTree not found. Skipping ast / ast+type modes."
    fi
else
    MODES=("${MODE}")
fi

# ── 실행 ──────────────────────────────────────────────────────
run_one_mode() {
    local m="$1"
    echo ""
    echo "========================================"
    echo " ConGra Preprocess: ${m}"
    echo "========================================"
    echo " Context lines: ${CONTEXT_LINES}"
    echo " Output:        ${OUTPUT_DIR}/dataset_${m}_ctx${CONTEXT_LINES}"
    echo "========================================"
    echo ""

    CMD="python3 preprocess.py \
        --data_dir ${DATA_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --context_lines ${CONTEXT_LINES} \
        --mode ${m} \
        --max_seq_len ${MAX_SEQ_LEN} \
        --tokenizer ${TOKENIZER} \
        --gumtree_bin ${GUMTREE_BIN}"

    if [ "${SKIP_STATS}" = true ]; then
        CMD="${CMD} --skip_stats"
    fi

    eval ${CMD}
}

for m in "${MODES[@]}"; do
    run_one_mode "${m}"
done

# ── 결과 요약 ─────────────────────────────────────────────────
echo ""
echo "========================================"
echo " Summary"
echo "========================================"
for m in "${MODES[@]}"; do
    dir="${OUTPUT_DIR}/dataset_${m}_ctx${CONTEXT_LINES}"
    if [ -d "${dir}" ]; then
        echo "  [OK] ${dir}"
    else
        echo "  [FAIL] ${dir}"
    fi
done
echo ""
