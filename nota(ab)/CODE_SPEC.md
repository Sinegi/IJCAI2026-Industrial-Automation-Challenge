# EQ / fsiq_csreg — 코드 기술명세서

FailureSensorIQ MCQA에 대한 **QLoRA + CSReg(구조 정규화) + Option-Content Listwise SFT + 학습된 NOTA Abstention** 파이프라인 명세. 2026-07-21 기준 실제 소스(`fsiq_csreg/`, `scripts/`, `configs/`)에 근거함.

> 이 문서의 **중심**은 §6-5 **학습된 abstain 로짓(predictor-rejector)** — NOTA("정답 없음") 문제 해결을 위한 최신 구현이다. 나머지 섹션은 그 맥락을 위한 파이프라인 개요.

---

## 1. 개요

산업 FMEA 5~8지선다(고장모드↔센서 관계)를 푸는 LLM을 학습/추론한다. 네 가지 학습 신호를 결합:

1. **LM SFT** — `<think>` 추론 + `<answer>`(label 또는 label+content) 생성.
2. **CSReg 구조 정규화** — asset 그래프 G*(edge/non-edge)를 residual projector로 재구성하는 4종 손실.
3. **Option-Content Listwise** — 모든 옵션을 anchor와 동시 비교하는 softmax 랭킹(극성 대칭).
4. **학습된 NOTA Abstention** — listwise softmax 안의 **abstain 로짓**(predictor-rejector) + G* 기반 **NOTA 합성행** + **표현 격리(A)** + **보정행(B)**.

추론은 **graph-free**: 학습된 LoRA 어댑터만 로드(G*/lexicon/게이트 미사용).

---

## 2. 저장소 구조

```
EQ/
├── fsiq_csreg/{__init__,core,trainer,public_data,graph_operator}.py
├── scripts/{prepare_data,train,evaluate,submit, ...진단}.py
├── configs/a100_40gb.yaml        # 단일 진실 소스
├── run.sh                         # prepare|train|eval|submit|smoke, 캐시 hdd3 고정
└── dataset/ , val.jsonl , test.jsonl
```
`core.py`가 데이터모델·전처리·그래프·프롬프트·손실·트레이너·예측기를 모두 포함. `__init__.py`는 `from .core import *`.

---

## 3. 데이터 모델 (core.py)

- `NormalizedMCQ` — 1문항. `options:OrderedDict[label→text], asset, direction∈{fm2sensor,sensor2fm,unknown}, polarity∈{positive,negative}, anchor, correct_labels, …`. 프로퍼티 `is_single_answer, answer_label`.
- `AssetGraph` — asset별 G*. `edges:set[(fm,sensor)], observed_nonedges, failure_modes, sensors`.
- `EntityLexicon`, `AssetOpsScenario`(추론 입력).

---

## 4. 데이터 파이프라인

### 4-1. 정규화·추론
`normalize_record` → `NormalizedMCQ`. `infer_direction`/`infer_polarity`(relevancy 토큰·질문 패턴), `extract_anchor`(lexicon 최장일치).

### 4-2. 그래프 G* (`build_train_graphs`) — 학습 라벨에서만
(fm,sensor) pair 투표: positive면 정답=edge, negative면 정답=non-edge. `positives/total ≥ 0.5`이면 `edge`, 아니면 `observed_nonedge`. **TRAIN-ONLY**.

### 4-3. 합성·증강
- `synthesize_single_answer_rows_from_graph`: edge로 positive/negative 단일정답행.
- **`synthesize_nota_rows_from_graph`** (§6-5c): non-edge만으로 옵션을 채워 gold=NOTA 행.
- **`synthesize_answerable_with_nota_rows_from_graph`** (§6-5e, 보정행 B): NOTA 옵션은 있지만 정답=실제 edge 옵션.
- `permute_options`: 옵션 셔플 증강.
- 학습은 prepared corpus(`data.prepared_dir` 재사용)에서 로드하며, **NOTA·보정 합성행은 train.py에서 G* 로드 후 주입**(prepare 재실행 불필요). 스모크 기준 base 6602 + NOTA 226 + 보정 229 = **7057행**.

---

## 5. 프롬프트 & 감독 신호

### 5-1. 사용자 프롬프트 (`build_marked_user_content`)
지시문 + Asset/Direction/Polarity + `<<ANCHOR>>` + 질문 + `<<OPTION_x>>` 조립, anchor/옵션 **문자 스팬** 반환. 말미 포맷 지시문은 `answer_style` 분기.

### 5-2. 어시스턴트 타깃 (`build_assistant_target`)
```
<think>{relation_reasoning}</think>
<answer>C</answer>                                    # answer_style=label
<answer>\n<label>C</label>\n<content>vibration sensor</content>\n</answer>   # label_content
```
NOTA 합성행은 label_content에서 `<content>None of the above</content>`가 타깃 → **생성 head가 NOTA를 낼 줄 학습**.

### 5-3. answer_style 전역 토글
`set_answer_style()/get_answer_style()` 모듈 전역. 학습·추론 일치, 추론은 어댑터 `csreg_config.json`에서 자동 복원.

### 5-4. 토큰화 (`tokenize_training_row`)
prompt 라벨 `-100` 마스킹, 문자→토큰 스팬 변환. 반환 필드에 listwise/abstain 감독 포함:
`… anchor_span, option_spans, option_mask, edge_targets, direction_id, answer_index, polarity_id,` **`nota_index`**(NOTA 옵션 위치 or -1, `nota_option_index`로 탐지).

### 5-5. 콜레이터 (`CSRegDataCollator`)
우패딩. 옵션 스팬 `max_options`까지 `[-1,-1]` 패딩. `answer_index, polarity_id,` **`nota_index`** 텐서화.

---

## 6. 모델 & 학습

### 6-1. 로딩 (train.py)
Base `Qwen/Qwen3-8B` 4bit NF4(bf16 compute) + gradient checkpointing. LoRA(r32/α64/drop0.05, q,k,v,o,gate,up,down). `attach_csreg_projector`로 `model.csreg_projection` 부착, **`attach_csreg_abstain_gate`로 `model.csreg_abstain_gate` 부착**(config `abstain_gate:true`).

### 6-2. Residual Projector (`make_residual_projector`)
`forward(x)=L2norm(x + 0.10·up(gelu(down(x))))`, down 64-rank(N(0,0.02)), up zero-init(초기 항등). `csreg_projection.pt` 저장.

### 6-3. 구조 손실 (`structural_losses`)
direction별 부모→자식 projector 사영. `signature, reconstruction, positive_alignment, nonedge_margin(relu(cos-0.15)²)`.

### 6-4. Option-Content Listwise (`listwise_loss`)
`mean_pool_spans`+L2정규화+공유 projector로 옵션 점수 `s_i`:
```
s_i = <projected(anchor), option_i>   (dir 0) | <projected(option_i), anchor> (dir 1) | <anchor,option_i> (unknown)
logit_i = sign·s_i/T          (positive sign=+1, negative sign=-1, T=0.1)
L = CrossEntropy(softmax(logits), gold)
```
옵션 4·8개 무관 동일, hard distractor에 직접 gradient. 유효 옵션<2/스팬불량이면 스킵(미분가능 0).

---

### 6-5. ★ 학습된 NOTA Abstention (predictor-rejector) — 핵심

**문제**: listwise softmax는 "가장 관련된 옵션 1개에 질량"을 학습 → 정답이 "해당 없음(NOTA)"인 문항과 정면 상충(재현 eval에서 NOTA family 0~5%). 근본 원인: "매칭 실패 → NOTA 선택" 신호 부재.

**해결(3요소)**: ① 표현공간에 학습되는 **abstain 로짓**(threshold 하드코딩 대신 end-to-end 학습) + ② 생성 head를 위한 **NOTA-positive 합성행**(C) + 관계형 회복을 위한 ③ **표현 격리(A)** + **보정행(B)**.

> **경험적 근거(1차 NOTA run, val tta=1)**: NOTA family 0~5% → **89~94%**, 전체 71.0%→**77.1%**, macro_family 57%→**80%**. 단 관계형(비-NOTA) 4개 family가 **균일 −10pp**(84.5%→74.4%, 옵션 수 전 구간). false-abstention 아님(비-NOTA엔 NOTA 옵션 없음, 예측이 마지막 슬롯 쏠림 없음). 원인=NOTA 행 CE가 실제 옵션 로짓을 끌어내려 **관계형 표현의 결단력을 전역적으로 약화**. → 개선 **A(격리)**·**B(보정행)** 도입.

#### (a) Abstain 게이트 = predictor-rejector 헤드 (`make_abstain_gate` / `attach_csreg_abstain_gate`)
```python
gate = nn.Linear(3, 1)      # zero-init (weight=0, bias=0 → 중립 시작)
```
- 입력 특징(실제 옵션 점수 통계): **`[max(s_real), mean(s_real), top1−top2]`**. **s_real은 `.detach()`** → 게이트 특징 경로가 표현을 왜곡하지 않음.
- 출력: **abstain 로짓** `b`. 실제 옵션 최고점이 낮거나 margin이 작으면 `b`↑ → NOTA 승리.
- `model.csreg_abstain_gate`로 부착. 학습 파라미터 **+4개**만 추가.

#### (b) listwise_loss 내 통합 — 격리(A) 포함 (`listwise_loss`, `nota_index`·`abstain_gate`)
전 옵션 인덱스 공간에서 동작(마스킹은 `-inf`):
```
scores  = 콘텐츠 점수 s (전 옵션)
logits  = sign·scores/T
if nota_index[b] 유효 and abstain_gate:
    real   = 유효옵션 \ {NOTA}
    s_real = scores[real].detach()                # 게이트 특징은 항상 detach
    logits[nota] = abstain_gate([max,mean,top1−top2]) / T
    # ── 개선 A: 표현 격리 ──
    if gold == nota:                              # NOTA-gold 행에서만
        logits[real] = logits[real].detach()      # 실제 옵션 로짓을 고정 기준으로
logits[¬valid] = -inf
L = CrossEntropy(logits, answer_index)            # gold는 원 인덱스 유지
```
- **gold 라우팅**: NOTA 행이면 `answer_index`=NOTA 슬롯 → 게이트 로짓이 정답 대상.
- **부호**: abstain 로짓 sign-agnostic(NOTA 문항은 positive).

#### (c) NOTA 합성행 (`synthesize_nota_rows_from_graph`) — 생성 head용
G* non-edge로 **모든 옵션이 anchor와 무관**한 문항, gold=`None of the above`(마지막 슬롯). `include_nota_synthetic`. 스모크 **226행**. LM head가 NOTA 라벨 생성을 학습.
- `is_nota_text` / `nota_option_index`: NOTA 옵션 정규화 탐지.

#### (d) ★ 개선 A — 표현 격리 (NOTA-gold 행)
**동기**: NOTA-gold 행의 softmax CE가 실제 옵션 로짓(=`sign·s/T`)을 끌어내려 → 공유 LoRA 표현이 "옵션들은 anchor와 덜 관련"으로 이동 → 관계형 랭킹 예리함 손실(경험적 −10pp).
**구현**: NOTA-gold 행에서 실제 옵션 로짓을 `.detach()`. CE가 **abstain 로짓만** 학습하고 관계형 표현엔 gradient 미전파. non-edge 분리는 `structural_losses.nonedge_margin`이 계속 담당.
**검증**: 단위테스트에서 NOTA-gold 행의 **hidden gradient = 0.0**, 일반 행 = 11.25, 게이트는 여전히 학습.

#### (e) ★ 개선 B — 보정행 (`synthesize_answerable_with_nota_rows_from_graph`)
**동기**: NOTA 옵션이 있는 훈련행이 전부 gold=NOTA면 모델이 "NOTA 옵션 존재→NOTA 정답" 편향 학습.
**구현**: NOTA 옵션이 **있지만 정답은 실제 edge 옵션**인 positive 행 생성. gold≠NOTA라 개선 A 격리가 안 걸려 **관계형 표현은 정상 학습**되고, 게이트는 "매치 있으면 abstain 금지"를 학습. `include_answerable_nota_synthetic`.
- 스모크: **229행**(NOTA 합성 226과 ~1:1 균형 → answerable/unanswerable 경계 보정). 검증: 보정행에서 표현 gradient 정상(16.9) + 게이트 학습.

#### (f) 왜 이 설계인가 (3안 비교)
| 기준 | 1.NOTA-class | 2.Answerability gate(2단) | **3.Abstain 로짓(채택)** |
|---|---|---|---|
| 기존 `s_i` 재사용 / 변경량 / 추론궁합 | 부분/중/보통 | ✗/대/나쁨 | **✓/소/좋음** |
| 이론 | 약 | 강(Abstain-QA) | 강(predictor-rejector, Mao 2024) |

3번이 "answerability gate를 softmax에 내장". 1번 절반(합성행 C)은 생성 head용, 2번의 answerable/unanswerable 균형은 보정행(B)으로 흡수.

#### (g) 추론에서의 역할
게이트는 **학습시 표현 정형화 auxiliary** — 추론(생성 기반 graph-free)은 게이트 미사용. NOTA를 실제로 "고르는" 것은 (c)로 학습된 **LM 생성**. 게이트는 `csreg_abstain_gate.pt` 저장(분석·향후 추론 오버라이드용).

#### (h) 검증 (단위 + 스모크, GPU 미충돌)
- 단위(CPU): NOTA/보정 합성행 성질, listwise+abstain 유한·미분, **A 격리(NOTA-gold hidden grad=0.0)**, **B 보정행(표현 grad 정상+게이트 학습)**, 하위호환.
- 스모크(48행, GPU1): NOTA 226 + 보정 229 주입, 게이트 부착(+4), 학습·저장, `csreg_config.abstain_gate:true`, skipped 0. (도는 submit=GPU0와 무충돌.)

---

### 6-6. 트레이너 (`trainer.py` MemoryEfficientCSRegTrainer)
- **final-norm forward hook**으로 마지막 hidden만 캡처(전 레이어 materialize 회피).
- **총손실**: `LM + Σλ_csreg·구조손실 + λ_listwise·listwise`. listwise는 `nota_index`·`model.csreg_abstain_gate`를 전달받아 abstention 포함.
- `save_model`: LoRA + `csreg_projection.pt` + **`csreg_abstain_gate.pt`**(게이트 존재 시) + `csreg_config.json`(모든 λ, `lambda_listwise`, `listwise_temperature`, `answer_style`, **`abstain_gate`**).
- `log`: `csreg/{lm,…,listwise,total}` 로그구간 평균.

### 6-7. TrainingArguments
bf16/tf32, grad checkpointing, `paged_adamw_8bit`, cosine, **`remove_unused_columns=False`**(커스텀 필드 보존 필수), save_steps/limit. projector·게이트는 peft 이후 부착돼 `requires_grad=True`로 optimizer 포함.

---

## 7. 추론 (`CSRegPredictor`)
graph-free. base+LoRA만 로드, 좌패딩, **TTA 순열 다수결**(옵션 내용 기준 투표). 답 추출 `ANSWER_PATTERNS`: `<label>X</label>`(listwise) → `<answer>X</answer>` → `\boxed{}` → … . 미생성 시 NLL 폴백(`answer_style` 맞춤 suffix). `predict_rows`에 라운드별 tqdm.

---

## 8. Config 레퍼런스 (`configs/a100_40gb.yaml`)

| 섹션 | 키 | 값 |
|---|---|---|
| model | base_model | `Qwen/Qwen3-8B` |
| lora | r/alpha/dropout | 32/64/0.05 |
| csreg | projector_rank/scale, λ×4, nonedge_margin | 64/0.10, 0.02/0.005/0.01/0.02, 0.15 |
| **listwise** | enabled / answer_style / lambda_listwise / temperature | true / label_content / 1.0 / 0.1 |
| **listwise (abstention)** | **include_nota_synthetic / nota_options_per_question / nota_per_anchor / abstain_gate** | **true / 5 / 1 / true** |
| **listwise (보정 B)** | **include_answerable_nota_synthetic / answerable_nota_per_anchor** | **true / 1** |
| training | output_dir | `/mnt/hdd3/user1/EQ_csreg_listwise_nota_AB_qwen3_8b` (hdd3) |
| | max_length/epochs/per_device·accum/lr | 768/2.0/4·16(eff 64)/8e-5 cosine |
| inference | batch_size/tta/max_new_tokens | 32/3/192 |

**Ablation 토글**: `abstain_gate/include_nota_synthetic/include_answerable_nota_synthetic`를 false로 → 순수 listwise baseline. 3점 비교: (baseline 71%) / (NOTA=합성행+게이트 77%) / (NOTA+A격리+B보정).

---

## 9. CLI (`run.sh`)
```bash
bash run.sh train  [--overwrite-output-dir]                 # QLoRA+CSReg+Listwise+NOTA
bash run.sh eval   --tta 1 [--output-dir outputs/validation_nota]
bash run.sh submit --adapter <path> --tta 1
bash run.sh train  --max-rows 48 --output-dir <hdd3dir> --overwrite-output-dir   # 스모크
```

---

## 10. 산출물 스키마
`final_adapter/`: LoRA + `csreg_projection.pt` + **`csreg_abstain_gate.pt`** + `csreg_config.json` + tokenizer.
`outputs/validation*/`: `metrics.json`, `metrics_by_{family,asset}.csv`, `val_predictions.csv`.
`submission.csv`: `id,answer`(QUOTE_ALL).

---

## 11. 현재 상태 / 다음

**실측 진행 (val 1242, tta=1)**

| 모델 | 전체 | macro_family | 관계형(비-NOTA) | NOTA |
|---|---|---|---|---|
| listwise (baseline) | 71.0% | 56.7% | 84.5% | 0~5% |
| NOTA (합성행 C + 게이트) | **77.1%** | **80.0%** | 74.4% (−10pp) | **89~94%** |
| NOTA + **A격리 + B보정** | 실학습 대기 | — | 관계형 회복 목표 | 유지 목표 |

- NOTA run이 전체·macro는 크게 개선했으나 관계형 −10pp 트레이드오프 → **A(격리)·B(보정행)** 로 관계형 회복을 노림(코드·CPU·GPU1 스모크 검증 완료, 실학습 대기).
- 실행: `bash run.sh train --overwrite-output-dir` → `eval --tta 1 --output-dir outputs/validation_nota_AB`. output_dir이 `_nota_AB_`라 이전 77% 어댑터(`_nota_`) 보존.
- 향후: A/B 효과 확인 후 NOTA 합성/보정 비율·난이도 튜닝, tta=3 재평가, 추론시 게이트 기반 abstain 오버라이드(anchor 필요 → graph-free 완화 트레이드오프) 검토.
