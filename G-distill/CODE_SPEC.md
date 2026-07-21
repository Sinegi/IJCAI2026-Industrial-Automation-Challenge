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
5. **G\*-Distillation (관계표 증류)** — closed-book 규칙상 추론에서 G*를 못 쓰므로 관계표를 가중치에 내재화: **RFD**(관계-사실 증류) + **SSA**(구조손실 λ 증폭). §6-6.

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
- **`synthesize_relation_fact_rows_from_graph`** (§6-6b, RFD): 관계표 전 쌍을 yes/no 사실로 증류(LM 손실만).
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
- **RFD 예외(§6-6b)**: `metadata['relation_fact']`인 행은 `option_mask=[0..]`, `edge_targets=[-1..]`, `answer_index=-1`, `nota_index=-1`로 덮어써 listwise·structural 감독을 끄고 **LM 손실만** 받게 한다.

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

### 6-6. ★ G\*-Distillation (관계표 증류) — closed-book 대응

**규칙 제약**: 본 트랙은 **엄격한 closed-book** — 추론 중 인터넷·검색·외부 DB·API·도구 사용 불가. 따라서 G\*를 **추론에서 조회할 수 없고**, 관계표는 반드시 **가중치 안에** 있어야 한다. G\*-Distillation은 그 목적의 기법군이다.

#### (a) 왜 필요한가 — 측정 근거
| 측정 | 값 | 함의 |
|---|---|---|
| G\* 룰 단독 val 정확도 (metadata 제공 시) | **1242/1242 = 100%** | 관계표는 사실상 완벽 |
| val 문항 중 train과 동일(질문+옵션) | 11/1242 = 0.9% | **행 누수 아님** |
| val anchor / 옵션 엔티티가 train에 등장 | 100% / 99.1% | 엔티티·관계표 공유 → 조합적 일반화 |
| 관계표 크기 (10 asset의 fm×sensor) | **1,342쌍** (edge 470 / non-edge 872) | 매우 작음 |
| **현재 학습행이 이미 노출하는 쌍** | **1,342/1,342 = 100%** | **커버리지 문제 아님** |
| 그럼에도 모델 정확도 | ~80% | → **보존(retention) 문제** |

즉 모델은 모든 쌍을 (평균 ~24회) 봤는데도 틀린다. 원인 가설: ① **신호 희석** — 생성 ~95토큰 중 정답은 사실상 1토큰이고 LM loss가 0.002까지 떨어진 것은 *추론 템플릿 암기*이지 관계 암기가 아님, ② **자산 간 간섭** — "Temperature"·"Vibration" 같은 센서명이 여러 asset에 등장하는데 edge 여부가 asset마다 다름.

> 참고: metadata 없는 test 조건을 val에서 시뮬레이션하면(텍스트만으로 복원) asset 100%·anchor 100%지만 **direction이 58%** 라 G\* 결정가능 63.8%로 떨어진다. 옵션 어휘(센서 집합 vs 고장모드 집합)로 방향을 판정하면 **direction 100%, 결정가능 90.8%, 결정분 정확도 100%**. 단 이 경로(G\* 조회)는 closed-book 위반이라 **채택하지 않음** — 대신 아래로 가중치에 내재화한다.

#### (b) RFD — Relation-Fact Distillation (`synthesize_relation_fact_rows_from_graph`)
G\*의 모든 (asset, fm, sensor) 쌍을 **조밀한 yes/no 사실**로 직접 드릴한다.
```
Q: For the {asset}, is the sensor "{sensor}" relevant for monitoring the failure mode "{fm}"?
   (sensor2fm 표현도 생성 → both_directions)
options: Yes / No   (순서 셔플 — Yes=A 지름길 차단)
target: <think>In the {asset} FMEA relation, "{sensor}" is (not) a relevant sensor for "{fm}".</think>
        <answer><label>A</label><content>Yes</content></answer>
```
- 생성량: **1,342쌍 × 2방향 = 2,684행**(gold Yes 940 / No 1744 = edge/non-edge 비율과 일치).
- **지도 토큰 48개**(기존 MCQA ~95개) → 사실당 신호 밀도 약 2배, **asset 조건부**로 명시.
- **옵션 감독 비활성화**: 행에 `metadata['relation_fact']=True`가 붙고 `tokenize_training_row`가 `option_mask=[0..]`, `edge_targets=[-1..]`, `answer_index=-1`, `nota_index=-1`로 설정 → listwise·structural이 이 행을 건너뛰고 **LM 손실만** 받는다("Yes"/"No"는 엔티티가 아니라 콘텐츠 스코어링이 무의미).
- config `relation_fact: {enabled, both_directions, repeats}`. `repeats`로 드릴 강도 조절.

#### (c) SSA — Structural Signal Amplification
구조손실이 총손실의 **0.3%**(0.0144/4.7)에 불과해 LM·listwise에 묻혀 있었다 → λ **10배** 상향.
```
lambda_signature 0.02→0.2   lambda_reconstruction 0.005→0.05
lambda_positive  0.01→0.1   lambda_nonedge       0.02→0.2
```
스모크 실측: 구조손실 기여 **0.0144 → 0.1441**, 총손실 대비 **0.3% → 3.1%**. (검산: lm 2.784 + listwise 1.730 + 구조 0.144 = 4.658 = 로그 total ✓)

#### (d) CDI — Consistent Decision Inference (예정, 미구현)
현재 최대 구조 모순: **학습은 스코어링(projector+abstain 임계)으로 결정하고, 추론은 텍스트 생성으로 결정**한다. 저장된 projector로 옵션 점수를 계산해 argmax + abstain 임계를 적용하면 학습·추론 결정 방식이 일치하고, 옵션 순서 민감성·생성 흔들림·NOTA 판단이 동시에 해결된다.
- ⚠️ 설계 주의: anchor 스팬 추출에 EntityLexicon을 쓰면 **외부 아티팩트 논란** → anchor 스팬 대신 **질문 전체 pooling**으로 설계해야 closed-book 안전.

#### (e) 학습 데이터 구성 (스모크 실측)
| 종류 | 행 수 |
|---|---|
| 기존 MCQA(prepared) | 6,602 |
| NOTA 합성 (§6-5c) | 226 |
| 보정 B (§6-5e) | 229 |
| **RFD 관계-사실** | **2,684** |
| **합계** | **9,741** (≈306 스텝, ~75분) |

#### (f) 검증
CPU 단위: RFD 2,684행·gold 분포·Yes 위치 50.6%·`relation_fact` 플래그, 토큰화에서 옵션 감독 비활성(mask 0 / targets -1 / index -1) + NOTA 행 회귀 정상. GPU 스모크(64행): 합성행 3종 주입, 게이트 부착, skipped 0, λ 상향 검산 일치, 저장 정상.

---

### 6-7. 트레이너 (`trainer.py` MemoryEfficientCSRegTrainer)
- **final-norm forward hook**으로 마지막 hidden만 캡처(전 레이어 materialize 회피).
- **총손실**: `LM + Σλ_csreg·구조손실 + λ_listwise·listwise`. listwise는 `nota_index`·`model.csreg_abstain_gate`를 전달받아 abstention 포함.
- `save_model`: LoRA + `csreg_projection.pt` + **`csreg_abstain_gate.pt`**(게이트 존재 시) + `csreg_config.json`(모든 λ, `lambda_listwise`, `listwise_temperature`, `answer_style`, **`abstain_gate`**).
- `log`: `csreg/{lm,…,listwise,total}` 로그구간 평균.

### 6-8. ⚠️ 프롬프트 분포 불일치 (val ≠ test) — 현재 최대 손실원

**문제**: 학습·val 프롬프트에는 `Asset / Direction / Question polarity / <<ANCHOR>>` 가 **정확히 채워져** 있지만, **test에는 `metadata`가 없어** 추론 경로에서 이 값들이 비거나 틀린다. `CSRegPredictor`는 빈 `EntityLexicon`을 쓰므로 앵커 추출도 사실상 불가.

| 필드 | val (metadata 有) | **test (submit.py 실제 경로)** |
|---|---|---|
| Asset | 10개 실제 자산 | **`unknown` 3048/3048 (100%)** |
| Anchor | 1242/1242 (100%) | **없음 2234/3048 (73.3%)** |
| Anchor 품질(있는 경우) | `temperature` (정확) | `in reciprocating internal combustion engine` — **자산명 오인, 오히려 방해** |
| Direction | 100% 확정 | **`unknown` 1395/3048 (45.8%)** |

즉 **지금까지의 모든 val 점수는 test에 없는 정보를 받은 낙관치**다. 모델은 학습 내내 정확한 asset/anchor/direction이 항상 있는 프롬프트로 훈련되었다.

#### (a) 정량화 — `evaluate.py --simulate-test`
val 행에서 `metadata`/`answer`를 제거해 `load_public_scenarios → normalize_scenario(빈 lexicon)` 경로로 프롬프트를 재구성(= submit.py와 동일)하고, **채점·그룹핑은 진짜 라벨/메타**로 한다. 옵션 순서가 동일해 라벨 매핑이 유지된다. `metrics.json`에 `simulate_test: true` 기록.

재현된 열화가 실제 test와 거의 일치: asset unknown 100%, anchor 없음 77.5%, direction unknown 41.9%.

**G\*-Distillation 측정 (val 1242, tta=1)**

| family | metadata 有 | **simulate-test** | Δ |
|---|---|---|---|
| positive_sensor_to_failure_nota | 94.8 | 79.2 | **−15.6** |
| positive_failure_to_sensor | 90.9 | 76.1 | **−14.8** |
| positive_sensor_to_failure | 89.7 | 78.1 | −11.6 |
| positive_failure_to_sensor_nota | 94.2 | 82.7 | −11.5 |
| negation_failure_to_sensor | 92.7 | 82.8 | −9.9 |
| negation_sensor_to_failure | 93.5 | 87.0 | −6.5 |
| **전체** | **91.87** | **80.19** | **−11.7** |

positive 계열이 가장 크게 무너진다(앵커 의존도 高). negation은 "NOT" 단서가 질문에 명시돼 덜 흔들린다.

#### (b) 실제 리더보드와의 대조 (test 42% 기준, ±약 2.4pp 노이즈)
| 모델 | val | test(LB) | 격차 |
|---|---|---|---|
| listwise | 71.01 | 69 | −2.0 |
| NOTA (C+게이트) | 77.13 | **67** | **−10.1** |
| NOTA+A+B | 80.27 | 75 | −5.3 |
| G\*-Distillation | 91.87 | (미제출) | simulate-test 추정 **≈80** |

- 격차가 2~10pp에 그치는 이유: **질문 본문에 자산·앵커가 자연어로 들어있어** 모델이 헤더 대신 본문을 읽어 부분적으로 복구한다.
- **NOTA run만 val 순위가 뒤집힘**(−10.1pp): abstention 판단이 앵커 유무에 민감 → 앵커가 없으면 "옵션들이 앵커와 무관한가"의 기준이 흐려져 과잉/과소 abstain. A+B는 보정행으로 균형이 잡혀 민감도가 낮다.
- test의 NOTA 문항 비중은 **10.0%**로 val(22.5%)의 절반 이하 → NOTA 개선은 LB에 덜 반영된다.

#### (c) simulate-test 프록시 보정 (검증 완료)
실제 test 점수를 아는 A+B(LB 75)로 프록시를 검증했다. test는 NOTA 비중이 **10.0%**(val 16.1%)이므로 family 정확도를 test 분포로 재가중해 비교한다.

| 모델 | simulate-test (val 분포) | **test 분포 보정** | 실제 test(LB) | 오차 |
|---|---|---|---|---|
| NOTA+A+B | 68.84% | **70.3%** | **75** | **−4.7pp** |
| G\*-Distillation | 80.19% | **80.1%** | 미제출 | — |

- 프록시는 **일관되게 저평가**한다(약 −5pp; LB가 test의 42%라 ±2.4pp 노이즈 포함 시 실제 −2~5pp).
- 그러나 **순위·상대격차는 보존**되므로, 오프셋(+≈4.7pp)을 적용하면 제출 없이 test 성능을 추정할 수 있는 실용 프록시로 쓸 수 있다.
- **G\*-Distillation 예상 test ≈ 85** (80.1 + 4.7, 범위 83~87) — 현행 최고(75) 대비 **+10pp**.

**프롬프트 열화에 대한 강건성 차이 (중요)**

| | metadata 有 → simulate-test | 낙폭 |
|---|---|---|
| A+B — NOTA | 68.8 / 70.2 → **40.6 / 55.8** | **−28.2 / −14.4** |
| G\*-Distill — NOTA | 94.8 / 94.2 → **79.2 / 82.7** | −15.6 / −11.5 |
| A+B — 관계형 | 80.6~84.4 → 70.8~76.5 | ≈ −9 |
| G\*-Distill — 관계형 | 89.7~93.5 → 76.1~87.0 | ≈ −9 |

A+B의 NOTA는 앵커가 없으면 **40%까지 붕괴**하지만 G\*-Distill은 **79~83%를 유지**한다. RFD가 관계를 **asset 조건부 자립 사실**로 각인시켜 metadata 스캐폴드 의존도를 낮춘 결과로, 점수 이상의 질적 개선이다.

#### (d) 대응 (진행 중)
1. ~~simulate-test 신뢰도 검증~~ — **완료**, (c) 참조. 오프셋 +≈4.7pp의 실용 프록시 확보.
2. **프롬프트 포맷 정합 학습** — 각 학습행에 **metadata-free 변형본**(Asset/Direction/Anchor 줄 없이 질문+옵션만)을 추가해 train 프롬프트 = test 프롬프트로 만든다. 앵커 스팬 감독이 필요한 원본 행은 유지하고 변형본은 **LM 손실만** 받게 한다(RFD와 동일 패턴). 11.7pp의 상당 부분 회복 기대.
3. ⚠️ 추론 시 lexicon/G\*로 앵커·자산을 복원하는 방식은 **closed-book 위반 위험**이라 채택하지 않는다. 모델이 질문 본문에서 스스로 읽도록 학습시키는 방향으로만 해결한다.

---

### 6-9. TrainingArguments
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
| csreg | projector_rank/scale, nonedge_margin | 64/0.10, 0.15 |
| **csreg λ (SSA, §6-6c)** | **signature/reconstruction/positive/nonedge** | **0.2 / 0.05 / 0.1 / 0.2** (10× 상향; 이전 0.02/0.005/0.01/0.02) |
| **relation_fact (RFD, §6-6b)** | **enabled / both_directions / repeats** | **true / true / 1** |
| **listwise** | enabled / answer_style / lambda_listwise / temperature | true / label_content / 1.0 / 0.1 |
| **listwise (abstention)** | **include_nota_synthetic / nota_options_per_question / nota_per_anchor / abstain_gate** | **true / 5 / 1 / true** |
| **listwise (보정 B)** | **include_answerable_nota_synthetic / answerable_nota_per_anchor** | **true / 1** |
| training | output_dir | `/mnt/hdd3/user1/EQ_csreg_B_qwen3_8b` (hdd3, G\*-Distillation run) |
| | max_length/epochs/per_device·accum/lr | 768/2.0/4·16(eff 64)/8e-5 cosine |
| inference | batch_size/tta/max_new_tokens | 32/3/192 |

**Ablation 토글**: `abstain_gate/include_nota_synthetic/include_answerable_nota_synthetic`를 false로 → 순수 listwise baseline. 3점 비교: (baseline 71%) / (NOTA=합성행+게이트 77%) / (NOTA+A격리+B보정).

---

## 9. CLI (`run.sh`)
```bash
bash run.sh train  [--overwrite-output-dir]                 # QLoRA+CSReg+Listwise+NOTA
bash run.sh eval   --tta 1 [--output-dir outputs/validation_nota]
bash run.sh eval   --tta 1 --simulate-test --output-dir outputs/..._simtest   # test 조건 재현(§6-8a)
bash run.sh eval   --tta 1 --simulate-test --adapter <path>                   # 특정 어댑터로 프록시 측정
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

| 모델 | val (metadata 有) | simulate-test | **test분포 보정** | test(LB 42%) | 결과 위치 |
|---|---|---|---|---|---|
| listwise (baseline) | 71.01% | — | — | 69 | `outputs/validation/` |
| NOTA (합성행 C + 게이트) | 77.13% | — | — | **67** | `outputs/validation_nota/` |
| NOTA + A격리 + B보정 | 80.27% | 68.84% | 70.3% | **75** (프록시 −4.7pp) | `outputs/validation_nota_AB/`, `_AB_simtest/` |
| **G\*-Distillation (RFD+SSA)** | **91.87%** | **80.19%** | **80.1%** | 미제출 · **추정 ≈85** | `outputs/validation_B/`, `_B_simtest/` |

- **G\*-Distillation이 전 family 동시 상승**(편차 15.6pp→5.1pp): NOTA 68.8/70.2 → **94.8/94.2**, 관계형 80.6~84.4 → **89.7~93.5**. "커버리지가 아닌 보존 문제"라는 진단(§6-6a)이 검증됨 — 시소가 사라짐.
- **그러나 최대 손실원은 이제 프롬프트 불일치(§6-8)**: metadata 유무만으로 **−11.7pp**. 어떤 하이퍼파라미터 튜닝보다 크다.
- **G\*-Distill은 프롬프트 열화에 강건**: A+B의 NOTA가 40%까지 붕괴하는 조건에서 79~83%를 유지(§6-8c). RFD가 관계를 asset 조건부 자립 사실로 각인시킨 효과.
- **진행 순서**: ① ~~simulate-test 검증~~ **완료**(오프셋 +≈4.7pp 프록시 확보) → ② **G\*-Distill 제출**(추정 ≈85, 현행 75 대비 +10pp) → ③ metadata-free 변형본 학습 구현 → ④ 재학습. 11.7pp를 절반만 회복해도 test **90 내외** 가시권.
- 이후: **CDI(§6-6d)** 스코어링 추론(질문 pooling, lexicon 미사용), RFD `repeats` 상향, tta=3 재평가, 약한 asset(steam turbine 79.3%, fan 83.8%) 보강.
