"""
ConGra Java merge conflict 데이터 전처리 스크립트.

raw_datasets/Java/ 에서 conflict region을 추출하고,
주변 컨텍스트와 함께 instruction-tuning 포맷으로 변환 후
프로젝트 단위 train/val/test split으로 저장한다.

Ablation modes:
  - baseline:  conflict region + context lines만
  - ast:       + GumTree edit script 요약 (base→a, base→b)
  - type:      + type context (imports, class/method signatures)
  - ast+type:  + 둘 다

Usage:
    python preprocess.py \
        --data_dir ./data/raw_datasets/Java \
        --output_dir ./data/processed \
        --context_lines 10 \
        --mode baseline

    # AST + type context 포함
    python preprocess.py \
        --mode ast+type \
        --gumtree_bin /path/to/gumtree
"""

import argparse
import json
import re
import subprocess
import shutil
import traceback
from pathlib import Path
from collections import defaultdict, Counter

from tqdm import tqdm
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer


# ── 프로젝트 단위 split 설정 ──────────────────────────────────────
SPLIT_CONFIG = {
    "train": [
        "spring-boot", "spring-framework", "jdk", "hadoop",
        "ghidra", "jenkins", "aosp", "micronaut",
    ],
    "val": ["dbeaver", "NewPipe"],
    "test": ["netty", "eclipse"],
}

# ── GumTree edit action 파싱 패턴 ─────────────────────────────────
# GumTree textdiff 출력 예: "Update MethodInvocation [123,456] ..."
GUMTREE_ACTION_RE = re.compile(
    r"^(Insert|Delete|Update|Move)\s+(\w+)", re.MULTILINE
)

# ── Java type context 추출 패턴 ───────────────────────────────────
IMPORT_RE = re.compile(r"^\s*import\s+[\w.*]+\s*;", re.MULTILINE)
CLASS_DECL_RE = re.compile(
    r"^\s*(?:(?:public|private|protected|abstract|final|static)\s+)*"
    r"(?:class|interface|enum|record)\s+\w+[^{]{0,200}\{",
    re.MULTILINE,
)
METHOD_SIG_RE = re.compile(
    r"^\s*(?:(?:public|private|protected|static|final|abstract|synchronized|native|default)\s+)*"
    r"(?:[\w<>\[\]?,]+\s+)+\w+\s*\([^)]{0,500}\)\s*(?:throws\s+[\w,\s]{1,200})?\s*\{",
    re.MULTILINE,
)


# ═══════════════════════════════════════════════════════════════════
# Core extraction functions
# ═══════════════════════════════════════════════════════════════════

def parse_region_file(region_path: str) -> list[tuple[int, int, int, int]]:
    """regions 파일 파싱 → (conflict_start, conflict_end, resolved_start, resolved_end) 리스트."""
    regions = []
    with open(region_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            m = re.match(r"\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)", line)
            if m:
                regions.append(tuple(int(x) for x in m.groups()))
    return regions


def extract_conflict_block(lines: list[str], start: int, end: int) -> str:
    """1-based line range [start, end]를 추출."""
    s = max(start - 1, 0)
    e = min(end, len(lines))
    return "".join(lines[s:e])


def add_context(lines: list[str], start: int, end: int, ctx: int) -> tuple[str, str, str]:
    """conflict region 앞뒤 ctx줄 컨텍스트 → (before, region, after)."""
    s = max(start - 1, 0)
    e = min(end, len(lines))
    before_s = max(s - ctx, 0)
    after_e = min(e + ctx, len(lines))
    return "".join(lines[before_s:s]), "".join(lines[s:e]), "".join(lines[e:after_e])


def has_conflict_markers(text: str) -> bool:
    return "<<<<<<<" in text


# ═══════════════════════════════════════════════════════════════════
# AST: GumTree edit script 추출
# ═══════════════════════════════════════════════════════════════════

def run_gumtree(gumtree_bin: str, file_a: Path, file_b: Path) -> str | None:
    """GumTree CLI로 두 파일의 edit script를 텍스트로 반환."""
    if not file_a.exists() or not file_b.exists():
        return None
    try:
        result = subprocess.run(
            [gumtree_bin, "textdiff", str(file_a), str(file_b)],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            return result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def summarize_edit_script(raw_output: str) -> str:
    """GumTree 원본 출력 → action type별 요약 문자열.

    예: "UPDATE MethodInvocation: 2, INSERT ImportDeclaration: 1, MOVE Block: 1"
    plan: "전체 edit script 그대로 넣으면 토큰 폭발 → action type 요약"
    """
    if not raw_output or not raw_output.strip():
        return "no changes"

    counts: Counter = Counter()
    for match in GUMTREE_ACTION_RE.finditer(raw_output):
        action = match.group(1).upper()
        node_type = match.group(2)
        counts[f"{action} {node_type}"] += 1

    if not counts:
        return "no changes"

    # 빈도 내림차순, 상위 10개만 (토큰 절약)
    top = counts.most_common(10)
    return ", ".join(f"{k}: {v}" for k, v in top)


def extract_ast_context(
    pair_dir: Path, java_name: str, gumtree_bin: str
) -> str:
    """base→a, base→b 두 방향의 edit script 요약을 생성."""
    base_file = pair_dir / "base" / java_name
    a_file = pair_dir / "a" / java_name
    b_file = pair_dir / "b" / java_name

    parts = []

    raw_base_a = run_gumtree(gumtree_bin, base_file, a_file)
    if raw_base_a:
        parts.append(f"base→a: {summarize_edit_script(raw_base_a)}")
    else:
        parts.append("base→a: unavailable")

    raw_base_b = run_gumtree(gumtree_bin, base_file, b_file)
    if raw_base_b:
        parts.append(f"base→b: {summarize_edit_script(raw_base_b)}")
    else:
        parts.append("base→b: unavailable")

    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════
# Type context: imports + class/method signatures
# ═══════════════════════════════════════════════════════════════════

def extract_type_context(
    merged_lines: list[str], conflict_start: int, conflict_end: int
) -> str:
    """merged 파일에서 type context를 추출.

    포함 항목:
      1. import 문 전체
      2. conflict를 감싸는 class/interface 선언
      3. conflict를 감싸는 method signature

    conflict 이전 텍스트만 검색하여 backtracking 위험을 줄인다.
    """
    # conflict 이전 텍스트만 잘라서 검색 → 긴 파일에서 성능 보장
    before_text = "".join(merged_lines[:max(conflict_start - 1, 0)])
    full_text = "".join(merged_lines)
    parts = []

    # 1) imports (전체 파일에서 추출)
    imports = IMPORT_RE.findall(full_text)
    if imports:
        parts.append("// Imports\n" + "\n".join(imports))

    # 2) class declarations - conflict 이전에서 마지막 매치
    class_matches = list(CLASS_DECL_RE.finditer(before_text))
    if class_matches:
        best_class = class_matches[-1].group(0).strip().rstrip("{").strip()
        parts.append(f"// Enclosing class\n{best_class}")

    # 3) method signatures - conflict 이전에서 마지막 매치
    method_matches = list(METHOD_SIG_RE.finditer(before_text))
    if method_matches:
        best_method = method_matches[-1].group(0).strip().rstrip("{").strip()
        parts.append(f"// Enclosing method\n{best_method}")

    return "\n".join(parts) if parts else ""


# ═══════════════════════════════════════════════════════════════════
# Prompt building (mode-aware)
# ═══════════════════════════════════════════════════════════════════

def build_prompt(
    before_ctx: str,
    conflict_region: str,
    after_ctx: str,
    ast_context: str | None = None,
    type_context: str | None = None,
) -> str:
    """mode에 따라 enriched prompt 생성."""
    prompt = "Below is a Java merge conflict. Resolve the conflict and output only the resolved code.\n\n"

    # type context (있으면 맨 위에 배치 — 모델이 타입 환경을 먼저 파악)
    if type_context and type_context.strip():
        prompt += f"{type_context}\n\n"

    # AST edit script 요약
    if ast_context and ast_context.strip():
        prompt += f"// Edit script summary\n{ast_context}\n\n"

    # 본문: context + conflict
    if before_ctx.strip():
        prompt += f"// Context before\n{before_ctx}\n"
    prompt += f"// Conflict\n{conflict_region}\n"
    if after_ctx.strip():
        prompt += f"// Context after\n{after_ctx}\n"

    prompt += "\n// Resolution\n"
    return prompt


# ═══════════════════════════════════════════════════════════════════
# Main processing
# ═══════════════════════════════════════════════════════════════════

def process_conflict_pair(
    pair_dir: Path,
    project: str,
    context_lines: int,
    mode: str,
    gumtree_bin: str | None,
) -> list[dict]:
    """하나의 conflict_files_N 디렉토리에서 모든 파일의 conflict region을 추출."""
    samples = []
    merged_dir = pair_dir / "merged"
    resolved_dir = pair_dir / "resolved"
    regions_dir = pair_dir / "regions"

    if not merged_dir.exists() or not resolved_dir.exists() or not regions_dir.exists():
        return samples

    use_ast = mode in ("ast", "ast+type")
    use_type = mode in ("type", "ast+type")

    for region_file in regions_dir.iterdir():
        if not region_file.name.endswith(".region"):
            continue
        java_name = region_file.name.replace(".region", "")
        merged_file = merged_dir / java_name
        resolved_file = resolved_dir / java_name

        if not merged_file.exists() or not resolved_file.exists():
            continue

        try:
            merged_lines = merged_file.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines(keepends=True)
            resolved_lines = resolved_file.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines(keepends=True)
        except Exception:
            continue

        # AST context는 파일 단위로 한 번만 추출 (region마다 동일)
        ast_ctx = None
        if use_ast and gumtree_bin:
            ast_ctx = extract_ast_context(pair_dir, java_name, gumtree_bin)

        regions = parse_region_file(str(region_file))
        for i, (cs, ce, rs, re_) in enumerate(regions):
            try:
                conflict_block = extract_conflict_block(merged_lines, cs, ce)
                if not has_conflict_markers(conflict_block):
                    continue

                before_ctx, conflict_region, after_ctx = add_context(
                    merged_lines, cs, ce, context_lines
                )
                resolved_region = extract_conflict_block(resolved_lines, rs, re_)

                # Type context는 region별로 추출 (감싸는 class/method가 다를 수 있음)
                type_ctx = None
                if use_type:
                    type_ctx = extract_type_context(merged_lines, cs, ce)

                prompt = build_prompt(
                    before_ctx, conflict_region, after_ctx,
                    ast_context=ast_ctx,
                    type_context=type_ctx,
                )
                text = prompt + resolved_region

                sample = {
                    "project": project,
                    "pair_id": pair_dir.name,
                    "file": java_name,
                    "region_idx": i,
                    "mode": mode,
                    "prompt": prompt,
                    "resolution": resolved_region,
                    "text": text,
                }
                if ast_ctx:
                    sample["ast_context"] = ast_ctx
                if type_ctx:
                    sample["type_context"] = type_ctx

                samples.append(sample)
            except Exception as e:
                print(f"    [WARN] {java_name} region {i}: {e}")
                continue

    return samples


def process_all(
    data_dir: Path,
    context_lines: int,
    mode: str,
    gumtree_bin: str | None,
) -> dict[str, list[dict]]:
    """전체 Java 프로젝트를 순회하며 샘플 추출 후 split별로 분류."""
    split_data = defaultdict(list)

    for project_dir in sorted(data_dir.iterdir()):
        if not project_dir.is_dir():
            continue
        project = project_dir.name

        split = None
        for s, projects in SPLIT_CONFIG.items():
            if project in projects:
                split = s
                break
        if split is None:
            print(f"  [WARN] Project '{project}' not in SPLIT_CONFIG, adding to train")
            split = "train"

        pair_dirs = [
            d for d in sorted(project_dir.iterdir())
            if d.is_dir() and d.name.startswith("conflict_files_")
        ]

        count = 0
        errors = 0
        for pair_dir in tqdm(pair_dirs, desc=f"  {project:20s}", leave=True):
            try:
                samples = process_conflict_pair(
                    pair_dir, project, context_lines, mode, gumtree_bin
                )
                split_data[split].extend(samples)
                count += len(samples)
            except Exception as e:
                errors += 1
                tqdm.write(f"    [ERROR] {pair_dir.name}: {e}")
                continue

        err_msg = f" ({errors} errors)" if errors else ""
        tqdm.write(f"  {project}: {count} samples → {split}{err_msg}")

    return split_data


def compute_token_stats(dataset: Dataset, tokenizer) -> dict:
    """토큰 길이 통계."""
    lengths = []
    for sample in dataset:
        ids = tokenizer(sample["text"], truncation=False)["input_ids"]
        lengths.append(len(ids))
    if not lengths:
        return {}
    lengths.sort()
    n = len(lengths)
    return {
        "count": n,
        "mean": sum(lengths) / n,
        "median": lengths[n // 2],
        "p90": lengths[int(n * 0.9)],
        "p95": lengths[int(n * 0.95)],
        "p99": lengths[int(n * 0.99)],
        "max": lengths[-1],
    }


def main():
    parser = argparse.ArgumentParser(description="ConGra Java conflict 전처리")
    parser.add_argument("--data_dir", type=str, default="./data/raw_datasets/Java")
    parser.add_argument("--output_dir", type=str, default="./data/processed")
    parser.add_argument("--context_lines", type=int, default=10,
                        help="conflict region 앞뒤로 포함할 컨텍스트 줄 수")
    parser.add_argument("--mode", type=str, default="baseline",
                        choices=["baseline", "ast", "type", "ast+type"],
                        help="전처리 모드 (ablation용)")
    parser.add_argument("--gumtree_bin", type=str, default="gumtree",
                        help="GumTree CLI 바이너리 경로 (ast/ast+type 모드에서 사용)")
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--tokenizer", type=str, default="deepseek-ai/deepseek-coder-1.3b-base")
    parser.add_argument("--skip_stats", action="store_true",
                        help="토큰 길이 통계 건너뛰기")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # AST 모드일 때 GumTree 설치 확인
    gumtree_bin = None
    if args.mode in ("ast", "ast+type"):
        if shutil.which(args.gumtree_bin):
            gumtree_bin = args.gumtree_bin
            print(f"GumTree found: {gumtree_bin}")
        else:
            print(f"[ERROR] GumTree not found at '{args.gumtree_bin}'.")
            print("  Install: https://github.com/GumTreeDiff/gumtree")
            print("  Or specify --gumtree_bin /path/to/gumtree")
            return

    print(f"Data dir:      {data_dir}")
    print(f"Mode:          {args.mode}")
    print(f"Context lines: {args.context_lines}")
    print()

    # 1. 전처리
    print("=" * 60)
    print("Step 1: Extracting conflict regions")
    print("=" * 60)
    split_data = process_all(data_dir, args.context_lines, args.mode, gumtree_bin)

    # 2. Dataset 생성
    print()
    print("=" * 60)
    print("Step 2: Building HuggingFace datasets")
    print("=" * 60)
    ds_dict = {}
    for split_name, samples in split_data.items():
        if samples:
            ds_dict[split_name] = Dataset.from_list(samples)
            print(f"  {split_name}: {len(samples)} samples")

    dataset = DatasetDict(ds_dict)

    # 3. 토큰 길이 통계
    if not args.skip_stats:
        print()
        print("=" * 60)
        print("Step 3: Token length statistics")
        print("=" * 60)
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
        stats = {}
        for split_name in dataset:
            s = compute_token_stats(dataset[split_name], tokenizer)
            stats[split_name] = s
            print(f"  {split_name}: {json.dumps(s, indent=2)}")

        for split_name in dataset:
            over = sum(
                1 for sample in dataset[split_name]
                if len(tokenizer(sample["text"], truncation=False)["input_ids"]) > args.max_seq_len
            )
            total = len(dataset[split_name])
            pct = over / total * 100 if total else 0
            if pct > 5:
                print(f"  [WARN] {split_name}: {over}/{total} ({pct:.1f}%) exceed max_seq_len {args.max_seq_len}")

        stats_path = output_dir / "token_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\n  Stats saved to {stats_path}")
    else:
        print("\nSkipping token stats (--skip_stats)")

    # 4. 저장 (mode별로 서브디렉토리 분리)
    print()
    print("=" * 60)
    print("Step 4: Saving dataset")
    print("=" * 60)
    save_dir = output_dir / f"dataset_{args.mode}_ctx{args.context_lines}"
    dataset.save_to_disk(str(save_dir))
    print(f"  Saved to {save_dir}")

    # 샘플 미리보기
    print()
    print("=" * 60)
    print("Sample preview (train[0])")
    print("=" * 60)
    if "train" in dataset and len(dataset["train"]) > 0:
        sample = dataset["train"][0]
        print(f"  Project: {sample['project']}")
        print(f"  File:    {sample['file']}")
        print(f"  Mode:    {sample['mode']}")
        print(f"  Prompt (first 500 chars):\n{sample['prompt'][:500]}")
        print(f"  Resolution (first 300 chars):\n{sample['resolution'][:300]}")

    print("\nDone!")


if __name__ == "__main__":
    main()
