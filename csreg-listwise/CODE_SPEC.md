# EQ / fsiq_csreg — 코드 기술명세서

FailureSensorIQ MCQA에 대한 **QLoRA + CSReg(구조 정규화) + Option-Content Listwise SFT** 파이프라인의 코드 명세. 2026-07-21 기준 실제 소스(`fsiq_csreg/`, `scripts/`, `configs/`)에 근거함.

---

## 1. 개요

산업 FMEA 기반 5~8지선다 문제(고장모드↔센서 관계)를 푸는 LLM을 학습/추론한다. 세 가지 학습 신호를 결합한다.

1. **LM SFT** — `<think>` 추론 + `<answer>` 생성 (`answer_style`에 따라 label 또는 label+content).
2. **CSReg 구조 정규화** — asset별 그래프 G*(edge/non-edge)를 hidden state 위 residual projector로 재구성하는 4종 손실.
3. **Option-Content Listwise** — 모든 선택지를 anchor와 동시에 비교하는 랭킹 손실(극성 대칭).

추론은 **graph-free**: 학습된 LoRA 어댑터만 로드하고 G*/lexicon은 쓰지 않는다.

---

## 2. 저장소 구조

```
EQ/
├── fsiq_csreg/              # 코어 패키지
│   ├── __init__.py          # `from .core import *`
│   ├── core.py              # 데이터모델·전처리·그래프·프롬프트·손실·트레이너·예측기 (~1850 lines)
│   ├── trainer.py           # MemoryEfficientCSRegTrainer (실제 학습에 사용)
│   ├── public_data.py       # HF FailureSensorIQ 다운로드→정규화→prepared corpus
│   └── graph_operator.py    # (선택) 그래프 연산 실험 모듈
├── scripts/
│   ├── prepare_data.py      # public corpus + G* 빌드 → prepared_dir
│   ├── train.py             # QLoRA+CSReg+Listwise 학습 엔트리
│   ├── evaluate.py          # val 평가 (accuracy/CI/macro, family·asset CSV)
│   ├── submit.py            # test → submission.csv
│   └── (진단) stage1_diagnostic.py, audit_graph.py, compare_ablation.py, analyze_error_patterns.py ...
├── configs/a100_40gb.yaml   # 단일 진실 소스 config
├── run.sh                   # prepare|train|eval|submit|smoke 래퍼 (env·캐시 hdd3 고정)
├── dataset/ , val.jsonl , test.jsonl (심볼릭)
└── csreg_sft.md , CODE_SPEC.md
```

`run.sh`는 모든 HF/캐시를 `/mnt/hdd3/user1/.cache`로 리다이렉트(홈 디스크 ~97% full 회피), conda `eq_vllm` 활성화, `CUDA_VISIBLE_DEVICES=0` 기본.

---

## 3. 데이터 모델 (core.py)

| 타입 | 위치 | 설명 |
|---|---|---|
| `NormalizedMCQ` | 102 | 정규화된 1문항. 필드: `id, question, options(OrderedDict[label→text]), passage, asset, relevancy, question_type, direction, polarity, anchor, correct_labels, reasoning, metadata`. 프로퍼티: `is_single_answer, answer_label, correct_mask, prompt_question`. |
| `AssetGraph` | 142 | asset별 G*. `failure_modes, sensors, edges:set[(fm,sensor)], observed_nonedges, vote_counts`. `edge(fm,sensor)` 조회. |
| `EntityLexicon` | 183 | asset별 failure_mode/sensor 어휘. anchor 추출용. |
| `AssetOpsScenario` | 86 | 추론 입력용 원시 시나리오(`id,text,metadata`). |

`direction ∈ {fm2sensor, sensor2fm, unknown}`, `polarity ∈ {positive, negative}`.

---

## 4. 데이터 파이프라인

### 4-1. 정규화 (`normalize_record`, 351)
원시 레코드 → `NormalizedMCQ`. 옵션 파싱(`_structured_options`), 정답 라벨 정규화(`_normalize_correct_labels` — bool 마스크/라벨/정답텍스트 매칭 모두 지원), reasoning 추출.

- **방향 추론** `infer_direction`(269): relevancy 토큰(`failure_to_sensor` 등) 우선, 없으면 질문 패턴(`which sensor…`/`which failure…`).
- **극성 추론** `infer_polarity`(297): relevancy가 `irrelevant/negation`으로 시작하거나 NOT/LEAST류 정규식 매칭 시 `negative`.
- **anchor 추출** `extract_anchor`(497)+`assign_anchors`: lexicon 최장일치.

### 4-2. 그래프 G* 빌드 (`build_train_graphs`, 570) — **학습 라벨에서만**
labeled + direction 확정 + anchor 있는 행만 사용. 각 (fm,sensor) pair에 대해:
- `pair_from_row_option`(551): direction에 따라 (anchor,option) 정렬.
- `option_edge_target`(563): positive면 정답=edge(1), **negative면 정답=non-edge**(정답이 아닌 옵션이 edge).
- asset별 pair 투표 → `positives/total ≥ min_vote_fraction(0.5)`이면 `edge`, 아니면 `observed_nonedge`.

> ⚠️ G*는 TRAIN-ONLY 산출물. val/test 라벨로 절대 재빌드 금지(`save_graph_bundle` 경고 포함).

### 4-3. 합성/증강
- `synthesize_single_answer_rows_from_graph`(703): G* edge로부터 단일정답 합성행 생성(`include_graph_synthetic`).
- `permute_options`(783): 옵션 내용 셔플로 위치편향 완화 증강(`permutation_copies`).
- `make_training_rows`(796): single + 합성 + permutation copies 취합.

### 4-4. prepared corpus (`public_data.prepare_public_corpus`, 166)
HF `FailureSensorIQ` 다운로드 → 정규화 → G* → `train_rows.jsonl` + `graph_bundle.json` + stats. `read_prepared_rows`로 로드. **현재 config는 `data.prepared_dir=/home/user1/SG/IQ/artifacts/prepared` 재사용**(prepare 스킵).

---

## 5. 프롬프트 & 감독 신호 구성

### 5-1. 사용자 프롬프트 (`build_marked_user_content`, 867)
지시문 + Asset/Direction/Polarity + `<<ANCHOR>>…<</ANCHOR>>` + 질문 + 옵션(`<<OPTION_x>>…`)을 조립하고 **anchor/각 옵션의 문자 스팬**을 함께 반환. 말미 지시문은 `answer_style`에 따라 분기(`label` → `<answer>LETTER</answer>`, `label_content` → label+content 블록).

### 5-2. 어시스턴트 타깃 (`build_assistant_target`, 910)
```
<think>{relation_reasoning}</think>
<answer>C</answer>                         # answer_style=label
# 또는
<answer>\n<label>C</label>\n<content>vibration sensor</content>\n</answer>   # label_content
```
`relation_reasoning`(842): direction/polarity에 맞춘 근거 문장 자동 생성(원본 reasoning 있으면 그대로).

### 5-3. answer_style 전역 토글 (824–839)
`set_answer_style() / get_answer_style()`. 모듈 전역 `_ANSWER_STYLE`. 학습(train.py)·추론(evaluate/submit)이 동일 값을 쓰도록 한 번 설정. 추론 측은 어댑터의 `csreg_config.json`에서 자동 복원.

### 5-4. 토큰화 (`tokenize_training_row`, 938)
- full = chat(prompt) + target + eos. prompt 구간 라벨은 `-100` 마스킹(어시스턴트만 학습).
- offset mapping으로 anchor/옵션 **문자 스팬 → 토큰 스팬** 변환(`char_span_to_token_span`).
- 반환 필드: `input_ids, attention_mask, labels, anchor_span, option_spans, option_mask, edge_targets, direction_id, asset_id, answer_index, polarity_id`.
  - `answer_index` = 정답 옵션의 위치, `polarity_id` = negative면 1 else 0 (listwise용).

### 5-5. 콜레이터 (`CSRegDataCollator`, 1009)
우패딩(Qwen 학습). input_ids/labels/mask 패딩 + 옵션 스팬을 `max_options`까지 `[-1,-1]` 패딩. `edge_targets(-1 pad, float), direction_id, asset_id, answer_index, polarity_id` 텐서화. (좌패딩 시 스팬 shift 보정 분기 포함.)

---

## 6. 모델 & 학습

### 6-1. 로딩 (train.py 93–151)
- Base: `cfg.model.base_model`(현재 `Qwen/Qwen3-8B`), 4bit NF4 double-quant, `bnb_4bit_compute_dtype=bf16`.
- `prepare_model_for_kbit_training` + gradient checkpointing(`use_reentrant=False`).
- LoRA(`peft.LoraConfig`): r=32, alpha=64, dropout=0.05, target=`q,k,v,o,gate,up,down_proj`.
- `attach_csreg_projector`로 `model.csreg_projection` 부착.
- transformers v4/v5 시그니처 차이 방어(`dtype` vs `torch_dtype`, 지원 안 되는 TrainingArguments 필드 필터).

### 6-2. Residual Low-Rank Projector (`make_residual_projector`, 1101)
```
down: Linear(hidden, rank=64, bias=False)  # init N(0,0.02)
up:   Linear(rank, hidden, bias=False)      # init 0 → 학습 초기 항등
forward(x) = L2normalize( x + scale(0.10) * up(gelu(down(x))) )
```
별도 `csreg_projection.pt`로 저장(추론엔 미사용).

### 6-3. 구조 손실 (`structural_losses`, 1135)
`mean_pool_spans`로 anchor/옵션 벡터 풀링 후 L2정규화. direction별 대칭:
- dir 0 (fm anchor→sensor options): anchor를 projector 통과, 옵션과 정합.
- dir 1 (fm options→sensor anchor): 옵션들을 projector 통과, 평균(옵션 수 불변) 후 anchor와 정합.

| 항 | 의미 |
|---|---|
| `signature` | projector 방향 정렬 `(-2·target·⟨projected,residual⟩)²` |
| `reconstruction` | edge 재구성 오차 `‖residual‖²` |
| `positive_alignment` | edge 쌍 `(1-cos)²` |
| `nonedge_margin` | non-edge `relu(cos - margin(0.15))²` |

### 6-4. Option-Content Listwise 손실 (`listwise_loss`, 1215) — 상세

#### (a) 동기 / 설계 목표
기존 LM SFT는 정답 **문자(letter)** 위치만 학습하고, CSReg 구조손실은 그래프 edge 쌍을 정합하지만 **한 문항 내 옵션들의 상대 순위**를 직접 최적화하지 않는다. Listwise는 "정답 옵션의 **내용(content)** 이 anchor와 가장 잘 정합하도록" 리스트 전체에 softmax 랭킹을 부과한다. 한 손실로 다음을 동시에 달성:
- 정답 문자 대신 **옵션 내용** 학습
- 4개·8개 보기 **동일 적용**(옵션 수 불변)
- **모든 선택지 동시 비교**(pairwise가 아닌 listwise)
- **Positive/Negative 대칭 처리**(부호 반전)
- **hard distractor에 직접 gradient**(softmax가 오답에 질량이 있으면 밀어냄)

이는 두 축(`answer_style: label_content` 생성 + `lambda_listwise` 손실)으로 나뉘며, §5-2의 label_content 타깃이 (a)를 LM 측에서, 본 손실이 표현공간 측에서 담당한다.

#### (b) 데이터 흐름 (감독 신호 출처)
Listwise는 새 두 필드에 의존하며 이는 학습 전처리에서 생성된다:
- `tokenize_training_row`(§5-4, 938): `answer_index = list(options).index(answer_label)`, `polarity_id = 1 if polarity=="negative" else 0`.
- `CSRegDataCollator`(§5-5, 1009): 두 필드를 배치 텐서 `answer_index[B], polarity_id[B]`로 묶음.
- `MemoryEfficientCSRegTrainer.compute_loss`(trainer.py): `inputs.pop("answer_index"/"polarity_id")` 후 캡처된 final hidden과 함께 `listwise_loss`에 전달.

anchor/옵션 표현은 **구조손실과 동일한 `mean_pool_spans` + L2정규화 + 동일 `model.csreg_projection`** 을 재사용한다(별도 파라미터 없음 → projector가 두 손실을 공유 학습).

#### (c) 스코어링 (방향별 geometry)
배치의 각 샘플 b에 대해, 유효 옵션 집합(`option_mask & option_valid`)만 골라 점수 벡터 `s ∈ R^K`를 만든다. projector 출력은 이미 L2정규화되어 있어 내적 ≈ 코사인.

```
dir 0 (fm2sensor, anchor=고장모드 → 옵션=센서):
    p = projector(anchor_b)                 # 부모(fm)를 자식(sensor) 공간으로 사영
    s_i = <p, option_i>                      for each valid option i

dir 1 (sensor2fm, anchor=센서 → 옵션=고장모드):
    p_i = projector(option_i)               # 각 옵션(부모 fm)을 사영
    s_i = <p_i, anchor_b>

dir unknown (방향 미상, 폴백):
    s_i = <anchor_b, option_i>               # projector 없이 원 코사인 유사도
```
방향 0/1은 구조손실의 부모→자식 사영 방향과 정확히 일치. unknown은 projector 기하를 못 쓰므로 순수 내용 유사도로 대체(그래도 "내용 매칭" 신호는 제공).

#### (d) 극성 대칭 & 온도 → 로짓 → 손실
```
sign = -1.0  if polarity_id[b] == 1 (negative)  else  +1.0
logits = (sign · s) / T                     # T = temperature (config 0.1)
gold_pos = 유효옵션 부분집합 내 answer_index의 위치
L_b = CrossEntropy(logits[None,:], [gold_pos])
L_listwise = mean_b(L_b)                     # 유효 샘플 없으면 0 텐서
```
- **Positive**: `softmax(s/T)` → 정답에 **높은 점수**가 정답. 정답 옵션의 anchor-정합을 끌어올림.
- **Negative(NOT/LEAST/irrelevant)**: `softmax(-s/T)` → 정답에 **낮은 점수**가 정답. "가장 무관한 옵션"을 정답으로 학습(부호만 뒤집어 동일 CE 재사용).
- **온도 T=0.1**: 코사인 s∈[-1,1]을 그대로 softmax하면 로짓 범위가 좁아 gradient가 약함. `/0.1`(≈ scale 10)로 분포를 날카롭게 해 실질적 랭킹 신호를 만든다.

#### (e) 엣지케이스 / 스킵 규칙
- anchor 스팬 무효 → 스킵.
- 유효 옵션 **2개 미만** → 랭킹 불가, 스킵.
- `answer_index < 0`(정답이 옵션에 없음) 또는 gold 옵션이 마스크 아웃 → 스킵.
- 스킵된 샘플은 `L_listwise`에 미기여. 전 배치가 스킵이면 `hidden.sum()*0.0` 형태의 **미분가능 0 텐서** 반환(그래프 유지, NaN 방지).

#### (f) 총손실 통합 & 로깅 (trainer.py)
```
components["listwise"] = listwise_loss(...)  if lambda_listwise>0 and 필드 존재  else 0텐서
total += lambda_listwise · components["listwise"]     # config 1.0
```
`csreg/listwise`가 로그구간 평균으로 리포트되고, `csreg_config.json`에 `lambda_listwise, listwise_temperature, answer_style` 저장 → 추론이 answer_style 자동 복원.

#### (g) 워크드 예시 (dir 0, positive, 3옵션)
```
anchor="bearing wear"(fm), 옵션 = [pressure, vibration(정답,idx1), temperature]
p = projector(anchor);  s = [<p,pres>, <p,vib>, <p,temp>] = [0.10, 0.55, 0.20]
logits = s/0.1 = [1.0, 5.5, 2.0];  softmax ≈ [0.010, 0.960, 0.030]
L_b = -log(0.960) ≈ 0.041     # vibration이 정답이므로 낮은 손실, 이미 잘 정렬됨
```
같은 문항이 negative("어느 센서가 무관한가")라면 `logits=-s/0.1`이 되어 **가장 낮은 s를 가진 옵션**에 질량이 실린다.

#### (h) 학습 관찰 (실측)
전체 학습에서 `csreg/listwise` 1.23(≈ln5, 초기 무작위) → **0.014**(epoch 2)로 수렴, 총손실 정합 검산 통과(lm + 1.0·listwise + Σλ·구조손실 = total). 단, 이 손실은 "관련 옵션 1개 선택"을 강화하므로 **정답 부재(NOTA)** 문제와 상충 → §11 참조.

> **간결 요약**: 유효 옵션 2개 미만/스팬 불량이면 스킵, 그 외엔 projector 기하로 `s`를 뽑아 `logits=sign·s/T`의 softmax CE. positive는 high-score 정답, negative는 low-score 정답, 옵션 수 무관 동일.

### 6-5. 트레이너 (`trainer.py` MemoryEfficientCSRegTrainer)
- **final-norm forward hook**으로 마지막 hidden만 캡처(`output_hidden_states=True`의 전 레이어 materialize 회피 — 40GB 절약).
- `model_accepts_loss_kwargs=False`(v5 grad-accum 스케일 오작동 방지).
- **총손실**:
  ```
  loss = LM_loss
       + λ_signature·signature + λ_reconstruction·reconstruction
       + λ_positive·positive_alignment + λ_nonedge·nonedge_margin
       + λ_listwise·listwise
  ```
- `log()`가 로그구간 평균 `csreg/{lm,signature,…,listwise,total}` 리포트.
- `save_model()`: LoRA + `csreg_projection.pt` + `csreg_config.json`(모든 λ + `nonedge_margin` + `lambda_listwise` + `listwise_temperature` + `answer_style`).

### 6-6. TrainingArguments (train.py 184–221)
bf16, tf32, grad checkpointing, `paged_adamw_8bit`, cosine, `remove_unused_columns=False`(커스텀 필드 보존 필수), `save_steps/save_total_limit`. 스텝 수는 `epochs × ceil(N/(batch·world·accum))`로 추정, warmup은 ratio 기반.

---

## 7. 추론 (`CSRegPredictor`, 1603)

- **graph-free**: base + LoRA 어댑터만 로드(`PeftModel.from_pretrained`, 1648). lexicon/G* 미사용.
- 토크나이저 **좌패딩**(generate), 4bit 옵션.
- **TTA** `tta_permutations`: 행별 옵션 순열 생성(`_permutations_for_row`), 각 순열로 배치 생성(`_generate_batch`, greedy `do_sample=False`), **옵션 내용 기준 다수결 투표**(위치편향 상쇄).
- **답 추출** `extract_answer_letter`(1553) + `ANSWER_PATTERNS`: `<label>X</label>`(listwise 포맷) → `<answer>X</answer>` → `\boxed{}` → `"answer":"X"` → 일반. 
- **NLL 폴백** `_score_label_candidates`: 유효 태그 미생성 시 각 라벨 suffix의 NLL 최소값 선택(`answer_style`에 맞춰 label 또는 label+content suffix).
- **진행바**: `predict_rows`에 라운드별 tqdm.

`generate_submission`(1798) → `id,answer` CSV(QUOTE_ALL). evaluate.py는 예측 vs gold로 accuracy/부트스트랩 95% CI/macro(asset·family) + family·asset CSV 산출.

---

## 8. Config 레퍼런스 (`configs/a100_40gb.yaml`)

| 섹션 | 키 | 현재값 | 비고 |
|---|---|---|---|
| model | base_model | `Qwen/Qwen3-8B` | |
| data | prepared_dir | `/home/user1/SG/IQ/artifacts/prepared` | 재사용, prepare 스킵 |
| | permutation_copies / include_graph_synthetic / graph_source_mode | 1 / true / single_only | |
| lora | r/alpha/dropout/target_modules | 32/64/0.05/7모듈 | |
| csreg | projector_rank/scale | 64/0.10 | |
| | λ_signature/reconstruction/positive/nonedge, nonedge_margin | 0.02/0.005/0.01/0.02, 0.15 | |
| **listwise** | enabled / answer_style / lambda_listwise / temperature | **true / label_content / 1.0 / 0.1** | `enabled:false` 또는 `answer_style:label`로 ablation |
| training | output_dir | `/mnt/hdd3/user1/EQ_csreg_listwise_qwen3_8b` | **hdd3**(홈 full 회피) |
| | max_length/epochs | 768/2.0 | |
| | per_device_batch/grad_accum | **4/16** (eff 64) | |
| | lr/scheduler/warmup_ratio/max_grad_norm/optim | 8e-5/cosine/0.05/0.5/paged_adamw_8bit | |
| inference | batch_size/tta_permutations/max_new_tokens | **32**/3/192 | |

---

## 9. CLI (`run.sh`)

```bash
bash run.sh prepare        # public corpus + G* → prepared_dir (현재 재사용이라 보통 스킵)
bash run.sh train          # QLoRA + CSReg + Listwise 학습
bash run.sh eval           # val 평가 → outputs/validation/
bash run.sh submit         # test → submission.csv
bash run.sh smoke          # 기존 어댑터로 20행 eval+submit 스모크
# 플래그 통과 예:
bash run.sh train  --max-rows 24 --output-dir <dir> --overwrite-output-dir
bash run.sh eval   --adapter <path> --tta 1
bash run.sh train  --disable-csreg          # CSReg λ 전부 0 (ablation)
```
- 어댑터 지정: `--adapter <path>` 또는 config `training.output_dir/final_adapter` 기본, `CSREG_ADAPTER` 환경변수(모듈 `predict()` 경로).
- listwise on/off: config `listwise.enabled` / `answer_style` (CLI 플래그 없음).

---

## 10. 산출물 스키마

| 파일 | 위치 | 내용 |
|---|---|---|
| `final_adapter/` | output_dir | LoRA(`adapter_model.safetensors`) + `csreg_projection.pt` + `csreg_config.json` + tokenizer |
| `checkpoint-*/` | output_dir | 중간 체크포인트(save_total_limit=2) |
| `training_manifest.json`, `train_results.json`, `skipped_rows.json` | output_dir | 학습 메타 |
| `val_predictions.csv` | outputs/validation | `id,gold,prediction,correct,asset,family,direction,polarity,anchor,n_options` |
| `metrics.json`, `metrics_by_family.csv`, `metrics_by_asset.csv` | outputs/validation | 지표 |
| `submission.csv` | EQ/ | `id,answer` (QUOTE_ALL) |

---

## 11. 알려진 한계 / 다음 작업

- **NOTA 붕괴**: listwise softmax가 "가장 관련된 옵션 1개에 질량"을 학습 → 정답이 "해당 없음"인 NOTA family에서 정면 상충(재현 eval에서 0~1%). 상세는 `csreg_sft.md` §2, §3-4.
- **다음**: NOTA를 가상 옵션으로 포함하는 **abstain-aware 랭킹**(anchor 최대 정합 < 임계 → NOTA 최상위) 또는 정답=NOTA 합성행 추가.
- **재현/데이터 무결성**: G*는 train-only. train=public FailureSensorIQ(HF), eval=competition val이므로 train/val 중복 여부 점검 권장.
