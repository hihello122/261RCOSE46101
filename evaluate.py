"""
ConGra Java merge conflict resolution - 평가 스크립트.

학습된 모델을 로드하여 test set에 대해 생성 및 평가 수행.

지표:
  1. Exact Match Rate - resolution 문자열 완전 일치율
  2. BLEU - 토큰 단위 BLEU score
  3. CodeBLEU - 코드 특화 BLEU (선택적, 라이브러리 설치 시)
  4. Edit Distance (Levenshtein) - 정규화된 편집 거리

Usage:
    python evaluate.py \
        --model_dir ./output/final \
        --dataset_dir ./data/processed/dataset \
        --split test \
        --max_new_tokens 512 \
        --batch_size 4

    # base 모델 (파인튜닝 전) 평가
    python evaluate.py \
        --model_name deepseek-ai/deepseek-coder-1.3b-base \
        --dataset_dir ./data/processed/dataset \
        --split test
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import PeftModel
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate merge conflict resolution model")

    # Model - 둘 중 하나만 지정
    parser.add_argument("--model_dir", type=str, default=None,
                        help="파인튜닝된 LoRA adapter 디렉토리")
    parser.add_argument("--model_name", type=str, default=None,
                        help="베이스 모델 이름 (파인튜닝 전 평가용)")

    # Data
    parser.add_argument("--dataset_dir", type=str, default="./data/processed/dataset")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="평가할 최대 샘플 수 (디버깅용)")

    # Generation
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0이면 greedy decoding")
    parser.add_argument("--batch_size", type=int, default=1)

    # Output
    parser.add_argument("--output_dir", type=str, default="./eval_results")
    parser.add_argument("--no_flash_attn", action="store_true")

    return parser.parse_args()


def normalize_code(code: str) -> str:
    """비교를 위한 코드 정규화: 공백/줄바꿈 차이 무시."""
    lines = code.strip().splitlines()
    lines = [line.rstrip() for line in lines]
    # 빈 줄 제거
    lines = [line for line in lines if line.strip()]
    return "\n".join(lines)


def compute_exact_match(pred: str, gold: str) -> bool:
    return normalize_code(pred) == normalize_code(gold)


def compute_edit_distance(pred: str, gold: str) -> float:
    """정규화된 Levenshtein 편집 거리 (0=동일, 1=완전히 다름)."""
    pred_norm = normalize_code(pred)
    gold_norm = normalize_code(gold)
    if pred_norm == gold_norm:
        return 0.0

    m, n = len(pred_norm), len(gold_norm)
    if m == 0 or n == 0:
        return 1.0

    # 메모리 효율적 DP
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            if pred_norm[i - 1] == gold_norm[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev

    return prev[n] / max(m, n)


def compute_bleu_score(pred: str, gold: str) -> float:
    """간단한 line-level BLEU-4 계산."""
    from collections import Counter

    def ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    pred_tokens = normalize_code(pred).split()
    gold_tokens = normalize_code(gold).split()

    if not pred_tokens or not gold_tokens:
        return 0.0

    # Brevity penalty
    bp = min(1.0, len(pred_tokens) / len(gold_tokens)) if gold_tokens else 0.0

    scores = []
    for n in range(1, 5):
        pred_ng = Counter(ngrams(pred_tokens, n))
        gold_ng = Counter(ngrams(gold_tokens, n))
        if not pred_ng:
            scores.append(0.0)
            continue
        clipped = sum(min(pred_ng[ng], gold_ng.get(ng, 0)) for ng in pred_ng)
        total = sum(pred_ng.values())
        scores.append(clipped / total if total > 0 else 0.0)

    if any(s == 0 for s in scores):
        return 0.0

    import math
    log_avg = sum(math.log(s) for s in scores) / 4
    return bp * math.exp(log_avg)


def load_model_and_tokenizer(args):
    """모델과 토크나이저를 로드."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    attn_impl = "eager" if args.no_flash_attn else "flash_attention_2"

    if args.model_dir:
        # LoRA adapter 로드
        adapter_config_path = os.path.join(args.model_dir, "adapter_config.json")
        with open(adapter_config_path, "r") as f:
            adapter_config = json.load(f)
        base_model_name = adapter_config.get(
            "base_model_name_or_path",
            "deepseek-ai/deepseek-coder-1.3b-base"
        )

        print(f"Loading base model: {base_model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            attn_implementation=attn_impl,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        print(f"Loading LoRA adapter: {args.model_dir}")
        model = PeftModel.from_pretrained(base_model, args.model_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    elif args.model_name:
        # 베이스 모델 직접 로드
        print(f"Loading base model: {args.model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            attn_implementation=attn_impl,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    else:
        raise ValueError("--model_dir 또는 --model_name 중 하나를 지정하세요.")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_resolution(model, tokenizer, prompt: str, args) -> str:
    """하나의 프롬프트에 대해 resolution을 생성."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.temperature == 0.0:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = args.temperature
        gen_kwargs["top_p"] = 0.95

    outputs = model.generate(**inputs, **gen_kwargs)
    # 입력 부분 제거하고 생성된 토큰만 디코딩
    input_len = inputs["input_ids"].shape[1]
    generated = outputs[0][input_len:]
    result = tokenizer.decode(generated, skip_special_tokens=True)
    return result


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. 데이터 로드 ───────────────────────────────────────────
    print("Loading dataset...")
    dataset = load_from_disk(args.dataset_dir)
    test_ds = dataset[args.split]
    if args.max_samples:
        test_ds = test_ds.select(range(min(args.max_samples, len(test_ds))))
    print(f"  {args.split}: {len(test_ds)} samples")

    # ── 2. 모델 로드 ─────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(args)

    # ── 3. 생성 및 평가 ──────────────────────────────────────────
    print(f"\nGenerating resolutions (max_new_tokens={args.max_new_tokens})...")

    results = []
    project_metrics = defaultdict(lambda: {"exact_match": [], "bleu": [], "edit_dist": []})

    for i, sample in enumerate(tqdm(test_ds, desc="Evaluating")):
        prompt = sample["prompt"]
        gold = sample["resolution"]
        pred = generate_resolution(model, tokenizer, prompt, args)

        em = compute_exact_match(pred, gold)
        bleu = compute_bleu_score(pred, gold)
        edit_dist = compute_edit_distance(pred, gold)

        result = {
            "idx": i,
            "project": sample["project"],
            "file": sample["file"],
            "exact_match": em,
            "bleu": bleu,
            "edit_distance": edit_dist,
            "prediction": pred,
            "gold": gold,
        }
        results.append(result)

        project = sample["project"]
        project_metrics[project]["exact_match"].append(int(em))
        project_metrics[project]["bleu"].append(bleu)
        project_metrics[project]["edit_dist"].append(edit_dist)

    # ── 4. 집계 ──────────────────────────────────────────────────
    total = len(results)
    overall = {
        "total_samples": total,
        "exact_match_rate": sum(r["exact_match"] for r in results) / total if total else 0,
        "avg_bleu": sum(r["bleu"] for r in results) / total if total else 0,
        "avg_edit_distance": sum(r["edit_distance"] for r in results) / total if total else 0,
    }

    per_project = {}
    for proj, metrics in project_metrics.items():
        n = len(metrics["exact_match"])
        per_project[proj] = {
            "count": n,
            "exact_match_rate": sum(metrics["exact_match"]) / n,
            "avg_bleu": sum(metrics["bleu"]) / n,
            "avg_edit_distance": sum(metrics["edit_dist"]) / n,
        }

    # ── 5. 출력 ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Overall Results")
    print("=" * 60)
    print(f"  Samples:        {overall['total_samples']}")
    print(f"  Exact Match:    {overall['exact_match_rate']:.4f}")
    print(f"  BLEU:           {overall['avg_bleu']:.4f}")
    print(f"  Edit Distance:  {overall['avg_edit_distance']:.4f}")

    print("\nPer-project Results:")
    for proj, m in sorted(per_project.items()):
        print(f"  {proj:20s}  n={m['count']:4d}  EM={m['exact_match_rate']:.3f}  "
              f"BLEU={m['avg_bleu']:.3f}  ED={m['avg_edit_distance']:.3f}")

    # ── 6. 저장 ──────────────────────────────────────────────────
    model_tag = "finetuned" if args.model_dir else "base"
    summary = {
        "model": args.model_dir or args.model_name,
        "split": args.split,
        "overall": overall,
        "per_project": per_project,
    }
    summary_path = output_dir / f"metrics_{model_tag}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nMetrics saved to {summary_path}")

    # 전체 prediction 저장
    preds_path = output_dir / f"predictions_{model_tag}.jsonl"
    with open(preds_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Predictions saved to {preds_path}")

    # 오답 분석용 - EM 실패 케이스 상위 N개
    failures = [r for r in results if not r["exact_match"]]
    failures.sort(key=lambda x: x["edit_distance"])
    print(f"\nClosest failures (lowest edit distance, top 5):")
    for r in failures[:5]:
        print(f"  [{r['project']}/{r['file']}] ED={r['edit_distance']:.3f} BLEU={r['bleu']:.3f}")
        print(f"    Gold: {normalize_code(r['gold'])[:100]}...")
        print(f"    Pred: {normalize_code(r['prediction'])[:100]}...")
        print()


if __name__ == "__main__":
    main()
