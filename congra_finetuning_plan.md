# ConGra Fine-tuning Plan

DeepSeek Coder + QLoRA를 이용한 Java merge conflict resolution 파인튜닝 계획.

## 프로젝트 개요

- **태스크**: Merge conflict resolution (Java)
- **데이터셋**: [ConGra](https://github.com/HKU-System-Security-Lab/ConGra) (HKU System Security Lab)
- **베이스 모델**: DeepSeek Coder 1.3B (→ 추후 6.7B 검토)
- **파인튜닝 방식**: QLoRA (4bit)
- **GPU**: NVIDIA L4 (24GB)

## 데이터셋: ConGra

### 기본 정보

- **원본 용도**: 벤치마크 (평가용). 파인튜닝용 포맷팅 스크립트 없음 → 직접 구성 필요
- **전체 규모**: 44,948 conflicts / 23,334 conflict files (34개 오픈소스 프로젝트)
- **언어**: C, C++, Java, Python
- **Java 비중**: 11/34 프로젝트 (전체의 약 30~35% 추정, ~13,000 conflicts)
- **버전**: `congra_full_datasets` (전체), `congra_tiny_datasets` (파일당 conflict 1개로 제한된 subset)

### 디렉토리 구조

```
raw_datasets/
└── <language>/
    └── <project>/
        └── <conflict_pair>/
            ├── a/           # A 버전 conflict 파일
            ├── b/           # B 버전 conflict 파일
            ├── base/        # 공통 조상
            ├── merged/      # diff3 포맷 merged
            ├── merged_without_base/
            ├── resolved/    # 정답 resolution
            └── regions/     # resolved conflict region의 line 범위

congra_{full,tiny}_datasets/
└── <language>/
    └── <classified_dir>/    # text, sytx, func, text+sytx 등 7개 카테고리
        ├── <conflict_file>
        └── meta_list.txt
```

### 분류 카테고리

`text`, `sytx`, `func`, `text+sytx`, `text+func`, `sytx+func`, `text+sytx+func`

### 다운로드

```bash
wget -c -O ConGra_dataset.tar.gz https://figshare.com/ndownloader/files/46967428
# md5: 869a312f577adcfe3a8654314a56d2a3
tar -xzvf ConGra_dataset.tar.gz -C data
```

## 기술 스택

- **AST 분석**: GumTree (conflict edit script 추출)
- **Type checking**: Eclipse JDT (resolution validity 검증)
- **ML Framework**: transformers, peft, trl, bitsandbytes
- **Attention**: Flash Attention 2

## 하드웨어 고려사항

### L4 GPU 특성

- VRAM 24GB
- BF16: ~121 TFLOPS
- 메모리 대역폭: 300GB/s (A100 대비 낮음 → 실제 학습 속도 영향)

### 모델 크기별 적재 가능성

| 모델 | 4bit 가중치 | QLoRA 최소 요구 | L4 적재 | 비고 |
|---|---|---|---|---|
| 1.3B | ~0.9GB | ~4GB | ✅ 여유 | 시작점 |
| 6.7B | ~4GB | ~10GB | ✅ 가능 | 빡빡함, gradient checkpointing 필수 |
| 33B | ~18GB | ~28GB | ❌ OOM | A100 40GB 이상 필요 |

## 학습 설정

### Batch size 전략

**L4는 메모리 대역폭 병목으로 batch를 키워도 속도 이득이 비례하지 않음:**

- batch 4 → 8: wall-clock 약 30~40% 단축 (좋음)
- batch 8 → 16: 추가 10~20% 단축 (미미)
- batch 16 → 32: 거의 동일하거나 오히려 느려짐

**Sweet spot**: `per_device_batch = 8~16`, `grad_accum = 1~2`

### 시퀀스 길이 결정

GumTree edit script + type context 포함 시 토큰 예산:

```
원본 conflict:              ~400 tokens
+ GumTree edit script:      +200~400 tokens
+ type context (imports,    +300~500 tokens
  class signatures):
─────────────────────────────────────────
합계:                       900~1300 tokens
```

→ **seq_len 2048 권장** (1024는 빠듯)

### Epoch 전략 (중요)

**10 epoch은 과적합 리스크 큼. 3~5 epoch + early stopping이 표준.**

데이터 크기별 권장:

| 데이터 크기 | 권장 epoch |
|---|---|
| <1K | 10~20 |
| 1K~10K | 3~5 |
| 10K~100K | 1~3 |
| >100K | 1 epoch도 길 수 있음 |

**Overfitting 리스크 포인트:**

1. Resolution 패턴이 반복적 ("take a", "take b", "concat")
2. 프로젝트별 관용구 암기 위험
3. 샘플 단위 random split 시 data leakage → **project-level held-out 필수**

**권장: early stopping 기반으로 자동 결정**

```python
num_train_epochs = 10              # 상한선
eval_strategy = "steps"
eval_steps = 200
save_strategy = "steps"
save_steps = 200
load_best_model_at_end = True
metric_for_best_model = "eval_loss"
early_stopping_patience = 3
```

### 예상 학습 시간 (Java full, L4 기준)

| 모델 | seq 2048, batch 4 기준 step/s | Java full 3 epoch |
|---|---|---|
| 1.3B | ~0.3 | 2~3시간 |
| 6.7B | ~0.08 | 10~15시간 |

Tiny 버전은 이 시간의 1/3~1/2.

## 추천 학습 설정 코드

### 1.3B 기본 설정

```python
from transformers import TrainingArguments
from peft import LoraConfig

# QLoRA config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Training args
training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,       # effective batch 16
    num_train_epochs=10,                 # early stopping으로 실제 제어
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    logging_steps=20,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="wandb",
)

# 모델 로드 시
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/deepseek-coder-1.3b-base",
    quantization_config=bnb_config,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
)

# SFTTrainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    peft_config=lora_config,
    max_seq_length=2048,
    packing=True,                        # 길이 편차 커서 필수
)
```

### 6.7B 스케일업 시 변경점

```python
# 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/deepseek-coder-6.7b-base",
    ...
)

# Training args 변경
per_device_train_batch_size = 2          # 4에서 2로 축소
gradient_accumulation_steps = 8          # effective batch 16 유지
gradient_checkpointing = True            # 필수
# 나머지 동일
```

## 전처리 파이프라인

### 입력 포맷 전략 선택지

1. **전체 conflict 파일 사용** (`merged_without_base`)
   - 장점: 컨텍스트 풍부
   - 단점: 길이 폭발, 1.3B에는 부담

2. **Conflict region만 추출** (`regions` 활용) ← **1.3B 권장**
   - 장점: 짧고 빠름
   - 단점: 주변 맥락 소실

3. **Region + 주변 N줄 컨텍스트** (절충안)

### GumTree edit script 사용 시 주의

전체 edit script 그대로 넣으면 토큰 폭발. **추려서 넣기:**

- Action type 요약 (INSERT/DELETE/UPDATE/MOVE 개수 + 대상 노드 타입)
- Move-edit 관계만 (resolution에 가장 중요한 신호)
- Type signature 변경만

이렇게 추리면 seq_len 1024로도 가능, 학습 속도 거의 2배.

### Validation split 전략

**프로젝트 단위로 split하세요** (샘플 단위 랜덤 split은 data leakage):

- Train: 8개 Java 프로젝트
- Val: 2개 프로젝트
- Test: 1개 프로젝트

## 평가 지표

eval_loss만 보지 말고 아래 지표를 함께 추적:

1. **Exact match rate** - resolution 문자열 일치율
2. **AST-level match** - GumTree로 생성 resolution과 ground truth AST 동등성 비교
3. **Compile-ability** - Eclipse JDT로 컴파일 가능 여부
4. **Type correctness** - Eclipse JDT type checker로 타입 일관성 검증

AST match나 compile rate가 plateau 치면 학습 중단. 보통 3~5 epoch.

## 단계별 실행 계획

### Phase 0: 환경 세팅 (0.5일)

- [ ] ConGra 데이터셋 다운로드 및 압축 해제
- [ ] Java 데이터 크기 실측 (`meta_list.txt` 카운트)
- [ ] 토큰 길이 분포 분석 (95 percentile 확인)
- [ ] GumTree/Eclipse JDT 연동 테스트

### Phase 1: 진단 실험 (1~2시간)

- [ ] Tiny + Java + func 카테고리로 시작
- [ ] 1.3B + seq_len 1024 + batch 4 + 10 epoch (early stopping)
- [ ] Loss curve와 exact match로 peak epoch 확인 (대개 3~5)

### Phase 2: 본 학습 (1.3B, 2~4시간)

- [ ] Full Java 데이터 사용
- [ ] Phase 1에서 찾은 peak epoch + 1~2로 설정
- [ ] AST match, compile rate까지 포함한 evaluation

### Phase 3: 스케일업 결정 (선택)

- [ ] 1.3B 결과 baseline 확보 후 6.7B로 업그레이드 검토
- [ ] 6.7B 본 학습 시 10~15시간 소요, 체크포인팅 주의

## 주의사항 & 팁

- **패킹 필수**: conflict 길이 편차 크기 때문에 `packing=True`로 padding 낭비 제거 (속도 20~40%↑)
- **Flash Attention 2**: L4에서도 효과 체감 (20~30%↑)
- **Paged optimizer**: OOM 방지용. `optim="paged_adamw_8bit"`
- **L4 세션 관리**: 10시간 넘는 학습은 체크포인트 자주 저장, 중단 시 재시작 가능하게
- **첫 run은 작게**: 파이프라인 검증이 먼저. 전처리 디버깅이 생각보다 오래 걸림

## 참고 링크

- [ConGra GitHub](https://github.com/HKU-System-Security-Lab/ConGra)
- [ConGra Paper (arXiv)](https://arxiv.org/abs/2409.14121)
- [DeepSeek Coder](https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-base)
- [GumTree](https://github.com/GumTreeDiff/gumtree)
- [Eclipse JDT](https://www.eclipse.org/jdt/)
