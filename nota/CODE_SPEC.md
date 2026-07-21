# EQ / fsiq_csreg — 코드 기술명세서

FailureSensorIQ MCQA에 대한 **QLoRA + CSReg(구조 정규화) + Option-Content Listwise SFT + 학습된 NOTA Abstention** 파이프라인 명세. 2026-07-21 기준 실제 소스(`fsiq_csreg/`, `scripts/`, `configs/`)에 근거함.

> 이 문서의 **중심**은 §6-5 **학습된 abstain 로짓(predictor-rejector)** — NOTA("정답 없음") 문제 해결을 위한 최신 구현이다. 나머지 섹션은 그 맥락을 위한 파이프라인 개요.

---

## 1. 개요

산업 FMEA 5~8지선다(고장모드↔센서 관계)를 푸는 LLM을 학습/추론한다. 네 가지 학습 신호를 결합:

1. **LM SFT** — `<think>` 추론 + `<answer>`(label 또는 label+content) 생성.
2. **CSReg 구조 정규화** — asset 그래프 G*(edge/non-edge)를 residual projector로 재구성하는 4종 손실.
3. **Option-Content Listwise** — 모든 옵션을 anchor와 동시 비교하는 softmax 랭킹(극성 대칭).
4. **학습된 NOTA Abstention** — listwise softmax 안의 **abstain 로짓**(predictor-rejector) + G* 기반 **NOTA 합성행**.

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
- **`synthesize_nota_rows_from_graph`** (신규, §6-5 참조): non-edge만으로 옵션을 채워 gold=NOTA 행.
- `permute_options`: 옵션 셔플 증강.
- 학습은 prepared corpus(`data.prepared_dir` 재사용)에서 로드하며, **NOTA 합성행은 train.py에서 G* 로드 후 주입**(prepare 재실행 불필요).

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

**해결(2축)**: ① 표현공간에 학습되는 **abstain 로짓**(threshold를 하드코딩 대신 end-to-end 학습) + ② 생성 head를 위한 **NOTA-positive 합성행**.

#### (a) Abstain 게이트 = predictor-rejector 헤드 (`make_abstain_gate` / `attach_csreg_abstain_gate`)
```python
gate = nn.Linear(3, 1)      # zero-init (weight=0, bias=0 → 중립 시작)
```
- 입력 특징(실제 옵션 점수 통계): **`[max(s_real), mean(s_real), top1−top2]`**.
- 출력: **abstain 로짓** `b`. 실제 옵션 최고점이 낮거나 top1-top2 margin이 작으면(anchor에 뚜렷이 맞는 게 없으면) `b`↑ → NOTA 승리.
- `model.csreg_abstain_gate`로 부착(projector와 동형, device/dtype 정렬). 학습 파라미터 **+4개**(weight 3 + bias 1)만 추가.

#### (b) listwise_loss 내 통합 (`listwise_loss`, `nota_index`·`abstain_gate` 인자)
전 옵션 인덱스 공간에서 동작(마스킹은 `-inf` 로짓으로):
```
scores  = 콘텐츠 점수 s (전 옵션)
logits  = sign·scores/T                         # 극성 반영
# --- NOTA 슬롯만 게이트 로짓으로 대체 ---
if nota_index[b] 유효 and abstain_gate:
    real   = 유효옵션 \ {NOTA}
    s_real = scores[real]
    feats  = [s_real.max(), s_real.mean(), top1−top2]
    logits[nota] = abstain_gate(feats) / T       # 콘텐츠 점수 대신 학습 로짓
logits[¬valid] = -inf                            # 패딩 옵션 마스킹
L = CrossEntropy(logits, answer_index)           # gold는 원 인덱스 유지
```
- **gold 라우팅**: NOTA 행이면 `answer_index`가 NOTA 슬롯 → 게이트 로짓이 정답 대상. 일반 행이면 실제 옵션이 정답, NOTA 슬롯 없음.
- **부호**: abstain 로짓은 sign-agnostic(NOTA 문항은 positive라 문제 없음).
- **gradient 흐름**: gold=NOTA일 때 CE가 (i) `b`↑(게이트가 낮은 점수→abstain 학습), (ii) 실제 옵션 `s_i`↓(non-edge 옵션을 anchor에서 밀어냄) → **공유 hidden/LoRA 표현이 "non-edge=저유사도"로 날카로워짐**.

#### (c) NOTA 합성행 (`synthesize_nota_rows_from_graph`) — 선행필수
G*의 non-edge로 **모든 옵션이 anchor와 무관**한 문항 생성, gold=`None of the above`(마지막 슬롯, competition 포맷 일치). direction fm2sensor/sensor2fm, polarity positive.
- `is_nota_text` / `nota_option_index`: `"none of the above"` 등 정규화 매칭으로 NOTA 옵션 탐지.
- train.py가 G* 로드 후 주입(`include_nota_synthetic`), 스모크에서 확인: **226행 생성**(전체 6,828). 이게 LM head가 NOTA 라벨을 생성하도록 직접 학습.

#### (d) 왜 이 설계인가 (3안 비교 결론)
| 기준 | 1.NOTA-class+리밸런싱 | 2.Answerability gate(2단) | **3.Abstain 로짓(채택)** |
|---|---|---|---|
| 기존 `s_i` 재사용 | 부분 | ✗(별도 head) | **✓** |
| 변경량 / 추론 궁합 | 중 | 대 / 나쁨 | **소 / 좋음** |
| 이론 | 약 | 강(Abstain-QA) | 강(predictor-rejector, Mao 2024) |

3번이 "answerability gate를 softmax에 내장"한 형태로 2번 이점을 최소 변경으로 획득. 1번의 유용한 절반(합성행)은 생성 head용으로 병행.

#### (e) 추론에서의 역할
게이트는 **학습시 표현 정형화 auxiliary** — 추론은 생성 기반(graph-free)이라 게이트를 직접 안 쓴다. NOTA를 실제로 "고르는" 것은 (c) 합성행으로 학습된 **LM 생성**이 담당하고, 게이트/abstain 로짓은 그 표현을 강화한다. 게이트 가중치는 분석·향후 추론-오버라이드용으로 `csreg_abstain_gate.pt` 저장.

#### (f) 검증 (단위 + 스모크)
- 단위: NOTA 합성행(gold=NOTA 마지막·distractor 전부 non-edge), listwise+abstain 유한·미분·**hidden과 게이트 양쪽 gradient**, 하위호환.
- 스모크(48행): 게이트 부착(+4 param) → 학습 → 저장 확인. 게이트 zero-init → 2스텝 후 ≈8e-5로 이동(실제 gradient 수신). `csreg_config.json`에 `abstain_gate:true`.

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
| training | output_dir | `/mnt/hdd3/user1/EQ_csreg_listwise_nota_qwen3_8b` (hdd3) |
| | max_length/epochs/per_device·accum/lr | 768/2.0/4·16(eff 64)/8e-5 cosine |
| inference | batch_size/tta/max_new_tokens | 32/3/192 |

**Ablation 토글**: `abstain_gate:false` + `include_nota_synthetic:false` → 순수 listwise baseline. → (baseline)/(+합성행)/(+합성행+게이트) 3점 비교.

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
- listwise 재현 baseline: val **71.0%**, 관계형 84~88%, **NOTA 0~5%**(붕괴).
- **본 구현(§6-5)** 으로 NOTA 학습 신호(합성행 + abstain 로짓) 추가 → 실학습·eval로 NOTA 개선 검증 예정(`bash run.sh train` → `eval --output-dir outputs/validation_nota`).
- 향후: 추론시 게이트 기반 abstain 오버라이드(anchor 필요 → graph-free 완화 트레이드오프), NOTA 합성행 비율/난이도 튜닝, threshold 캘리브레이션.
