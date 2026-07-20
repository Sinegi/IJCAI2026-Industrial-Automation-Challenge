# Industrial Automation Challenge - Track 1

> 원문: [Kaggle 대회 개요
> ](https://www.kaggle.com/competitions/industrial-automation-challenge-track-1/overview)데이터셋: [Github](https://github.com/IBM/AssetOpsBench/tree/ijcai_2026_competition)
> 번역 확인일: 2026-07-19

## 개요

산업 자동화에는 센서 원격 측정 데이터(sensor telemetry), 설비의 물리적 특성, 고장 모드, 유지보수 의사결정 사이의 복잡한 관계를 추론할 수 있는 AI 모델이 필요합니다. 대규모 언어 모델과 추론 모델은 범용 추론 과제에서 가능성을 보여 왔지만, 실제 산업 유지보수 상황에서 구조적이고 물리 법칙에 기반한 추론을 수행하는 능력은 아직 충분히 검증되지 않았습니다.

## 업데이트

**(7월 1일)** 오픈 소스 모델 사용 요건을 준수했는지 확인하기 위해 모든 팀은 예측 파일과 함께 추론에 사용한 노트북 또는 코드를 제출해야 합니다. 이는 일부 제출물이 최상위급 독점형 LLM을 사용해 답을 생성했을 수 있다는 우려에 따른 조치입니다. **2026년 7월 15일**부터 제출된 노트북을 완전히 비공개인 테스트 세트로 평가합니다. 이 평가를 위한 별도의 리더보드는 대회 웹사이트에 공개될 예정입니다.

**(7월 1일)** 대회 포털에 설명된 것처럼 대상 모델에는 LLaMA, DeepSeek-R1 및 이와 유사한 오픈 소스 LLM이 포함됩니다. 이 오픈 소스 모델 요건을 충족하지 않는 제출물은 실격 처리됩니다.

**(7월 1일)** 모델 메타데이터와 추론 과정 기록(reasoning traces)을 포함하도록 제출 파일 형식이 변경되었습니다. 평가 지표는 그대로이며, 새로 추가된 열은 검증 목적으로 사용됩니다.

**(7월 1일)** 테스트 세트가 확대되었습니다. 현재 리더보드 점수는 이전 테스트 세트를 기준으로 합니다. 새로운 문제로 점수를 받으려면 다시 제출해야 하며, 최종 비공개 평가를 위해 7월 1일 이후에 제출한 유효한 결과를 선택해야 합니다.

**(6월 28일)** 테스트 세트 업데이트: **2026년 7월 1일**, 테스트 문제에 250개 이상의 예시가 추가됩니다. 테스트 세트가 변경되므로 모든 팀은 7월 1일 이후 예측 결과를 다시 제출해야 합니다. 원활한 재제출을 위해 2026년 7월 1일부터 일일 제출 횟수 제한을 10회로 늘립니다.

**(6월 28일)** 최종 수상 결과 발표 후 주최 측이 연락할 수 있도록 아래 양식에 연락처를 작성해 주세요. 👉 [연락처 정보 양식](https://forms.gle/QyB8sD4S951fzGb39)

**(6월 28일)** 이 트랙에는 상위 3개 우승 팀을 위한 총 250달러의 상금이 배정되어 있습니다. 상금 배분에 관한 자세한 내용은 대회 종료 1주 전에 공개될 예정입니다.

## 트랙 1: 모델 내부 추론

**Industrial Automation Challenge - Track 1**은 AI 모델이 내부 파라미터에 저장된 지식만을 사용하여 물리 법칙에 기반한 산업 추론을 수행할 수 있는지를 평가합니다.

참가자는 설비, 센서, 고장 모드, FMEA(고장 형태 및 영향 분석) 형식의 맥락, 진단 단서가 포함된 유지보수 및 고장 분석 객관식 문제에 답해야 합니다. 각 제출물에는 선택한 답을 포함해야 하며, 요구되는 경우에는 모델이 해당 답을 선택한 이유를 설명하는 간결한 진단 근거도 함께 제시해야 합니다.

이 트랙은 엄격한 폐쇄형(closed-book) 환경으로 운영됩니다. 추론 중에는 인터넷 접속, 검색 시스템, 외부 데이터베이스, API, 도구 또는 에이전트형 워크플로를 사용할 수 없습니다. 목표는 독립 실행형 모델이 외부 도움 없이 산업 설비의 물리 원리와 고장 메커니즘을 얼마나 잘 추론하는지 측정하는 것입니다.

---

# 데이터셋

- 산업 설비별 ‘고장 모드 ↔ 관련 센서’ 관계를 여러 형태의 객관식 문제로 변환한 데이터셋 => G를 만들어야하나
- FMEA/FMECA에 기반한 지식형 질의응답 데이터
  - FMEA = Failure Mode and Effects Analysis: 어떤 부품이나 설비가 어떤 방식으로 고장 날 수 있고, 그 고장이 시스템에 어떤 영향을 주는가
  - FMECA = Failure Mode, Effects, and Criticality Analysis: 그 고장이 얼마나 치명적인가/우선순위가 높은가

### `question`

실제로 풀어야 하는 질문입니다.

질문에는 보통 다음 정보가 들어 있습니다:

```
설비 종류
+ 고장 또는 센서
+ 질문 방향
+ 긍정 또는 부정 조건
```

Q. For electric motor, which failure event is not pertinent if the sensor temperature registers an abnormal reading?

=>

설비: electric motor
출발점: Temperature 센서 이상
방향: 센서 → 고장
조건: 관련 없는 고장 선택

### `options` 선택지 수

| 선택지 수 | Val | Test |
| --------- | --- | ---- |
| 4개       | 112 | 769  |
| 5개       | 265 | 669  |
| 6개       | 453 | 942  |
| 7개       | 259 | 455  |
| 8개       | 153 | 213  |

따라서 모델이 무조건 `A`~`G`를 기대하면 안 됩니다. 문제마다 실제로 존재하는 선택지 문자를 확인해야 합니다.

### `metadata`

검증 문제의 구조를 설명하는 보조 정보입니다.

```JSON
"metadata": {
  "asset_class": "electric motor",
  "family": "negation_sensor_to_failure",
  "anchor": "Temperature",
  "n_options": 5
}
```

* `asset_class`: 대상 산업 설비
* `family`: 문제 유형
* `anchor`: 질문의 출발점이 되는 고장 또는 센서
* `n_options`: 선택지 개수

검증 데이터에는 10종류의 설비가 있습니다.

| 설비                                     | 문항 수 |
| ---------------------------------------- | ------- |
| Power transformer                        | 380     |
| Reciprocating internal combustion engine | 120     |
| Compressor                               | 110     |
| Aero gas turbine                         | 108     |
| Industrial gas turbine                   | 104     |
| Electric motor                           | 98      |
| Electric generator                       | 94      |
| Steam turbine                            | 82      |
| Fan                                      | 80      |
| Pump                                     | 66      |

## 문제 유형

검증 데이터의 `metadata.family`를 기준으로 총 6가지 유형이 있습니다. 문제는 크게 **고장에서 센서를 찾는 방향**과 **센서에서 고장을 찾는 방향**으로 나뉘며, 여기에 긍정·부정·`None of the above` 조건이 결합됩니다.

| 유형                                | 질문 방향 및 조건                         |         문항 수 |
| ----------------------------------- | ----------------------------------------- | --------------: |
| `positive_failure_to_sensor`      | 고장 → 관련 센서                         |             330 |
| `positive_sensor_to_failure`      | 이상 센서 → 관련 고장                    |             320 |
| `negation_failure_to_sensor`      | 고장 → 관련 없는 센서                    |             192 |
| `negation_sensor_to_failure`      | 이상 센서 → 관련 없는 고장               |             200 |
| `positive_failure_to_sensor_nota` | 고장 → 센서, 정답은`None of the above` |             104 |
| `positive_sensor_to_failure_nota` | 센서 → 고장, 정답은`None of the above` |              96 |
| **합계**                      |                                           | **1,242** |

### 1. 고장 → 관련 센서

`positive_failure_to_sensor`는 특정 고장이 발생했을 때 어떤 센서를 우선 확인해야 하는지 묻습니다.

```text
냉각 시스템 고장
→ 어떤 센서가 가장 직접적으로 반응하는가?
→ Engine temperature
```

고장이 물리적으로 어떤 변화를 일으키는지 생각한 뒤, 그 변화를 직접 측정하는 센서를 선택해야 합니다.

### 2. 이상 센서 → 관련 고장

`positive_sensor_to_failure`는 특정 센서에서 이상값이 관측됐을 때 어떤 고장을 의심해야 하는지 묻습니다.

```text
Ultrasound 센서 이상
→ 어떤 고장과 관련 있는가?
→ Core looseness
```

하나의 센서가 여러 고장과 관련될 수 있으므로, 현재 선택지 중 해당 센서와 가장 직접적인 관계가 있는 고장을 골라야 합니다.

### 3. 고장 → 관련 없는 센서

`negation_failure_to_sensor`는 특정 고장을 감지하는 데 적절하지 않은 센서를 고르는 문제입니다.

```text
Bearing damage
→ 다음 중 이 고장을 나타내지 않는 센서는?
```

`does not indicate`, `not relevant`, `except` 같은 부정 표현을 먼저 확인해야 합니다. 관련 센서가 아니라 **관련성이 가장 낮은 센서**를 선택합니다.

### 4. 이상 센서 → 관련 없는 고장

`negation_sensor_to_failure`는 특정 센서의 이상값과 관련 없는 고장을 고르는 문제입니다.

```text
Temperature 센서 이상
→ 다음 중 관련 없는 고장 모드는?
```

`not pertinent`, `could not cause`, `all but one`도 관련 없는 선택지를 요구하는 표현입니다. 긍정 문제로 잘못 읽으면 정반대 답을 선택하게 됩니다.

### 5. 고장 → 센서, 정답 없음

`positive_failure_to_sensor_nota`는 고장과 관련된 센서를 찾는 문제지만, 제시된 구체적인 센서가 모두 부적절합니다. 따라서 정답은 `None of the above`입니다.

```text
특정 고장
→ 선택지의 센서를 하나씩 검토
→ 적절한 센서가 하나도 없음
→ None of the above
```

### 6. 이상 센서 → 고장, 정답 없음

`positive_sensor_to_failure_nota`는 센서와 관련된 고장을 찾는 문제지만, 제시된 고장 모드가 모두 부적절합니다. 이 경우에도 `None of the above`를 선택합니다.

`None of the above`가 선택지에 있다고 항상 정답인 것은 아닙니다. 먼저 다른 선택지를 모두 평가한 뒤, 적절한 답이 없을 때만 선택해야 합니다.

| 구분         | `val`         | `test`             |
| ------------ | --------------- | -------------------- |
| 문항 수      | 1,242개         | 3,048개              |
| `answer`   | 있음            | 없음                 |
| `metadata` | 있음            | 없음                 |
| 주요 용도    | 모델 평가·개선 | 최종 답안 생성·제출 |

`metadata`를 이용하면 다음처럼 세부 성능도 확인할 수 있습니다.

```
전체 정확도: 61%

긍정 문제: 75%
부정 문제: 42%
None of the above: 38%
전동기 문제: 68%
변압기 문제: 55%
```

```Python
tokenizer = load_tokenizer()
model = load_model()


def predict(scenario):
    # 이미 로드된 모델 재사용
    ...
    return {"answer": answer}
```

## `scenario`는 무엇인가?

`scenario`는 문제 하나를 담은 Python 객체입니다.

```
def predict(scenario):
    print(scenario.id)
    print(scenario.text)
    print(scenario.options)
    print(scenario.metadata)

    return {"answer": "A"}
```

각 속성의 의미는 다음과 같습니다.

* `scenario.id`: 문제 ID
* `scenario.text`: 모델에게 전달할 문제 본문
* `scenario.options`: `A`, `B`, `C` 등의 선택지
* `scenario.metadata`: 문제에 딸린 추가 정보
* `scenario.to_dict()`: 객체 전체를 일반 Python 딕셔너리로 변환

---

## 추천 진행 순서

### 1. Zero-shot 기준점 만들기

먼저 오픈 소스 모델 하나에 정답 예시나 추가 학습 없이 검증 문제 1,242개를 풀게 하고 정확도를 계산합니다. 이를 **Zero-shot 기준점(baseline)**이라고 합니다.

여기서 `zero-shot`은 모델에게 정답이 포함된 예시를 하나도 보여주지 않는다는 뜻입니다. 검증 데이터의 `answer`는 모델에게 입력하지 않고, 모델의 예측이 끝난 다음 채점할 때만 사용합니다.

```text
문제 설명 + 질문 + 선택지
          ↓
   기존 모델로 예측
          ↓
     정답과 비교
```

이 기준점을 먼저 만들어야 이후에 프롬프트 변경, Few-shot 예시 추가, 모델 변경 또는 LoRA 학습이 실제로 성능을 높였는지 비교할 수 있습니다.

### 2. 유형별 정확도 계산하기

전체 정확도뿐 아니라 검증 데이터의 `metadata.family`별 정확도를 계산합니다. 이 데이터에는 다음 유형이 있습니다.

| 유형                                | 의미                                         |
| ----------------------------------- | -------------------------------------------- |
| `positive_failure_to_sensor`      | 고장과 관련된 센서 선택                      |
| `positive_sensor_to_failure`      | 이상 센서와 관련된 고장 선택                 |
| `negation_failure_to_sensor`      | 고장과 관련 없는 센서 선택                   |
| `negation_sensor_to_failure`      | 이상 센서와 관련 없는 고장 선택              |
| `positive_failure_to_sensor_nota` | 적절한 센서가 없어`None of the above` 선택 |
| `positive_sensor_to_failure_nota` | 적절한 고장이 없어`None of the above` 선택 |

유형별 정확도는 모델이 **무엇을 왜 틀리는지** 찾기 위한 디버깅 도구입니다. 예를 들어 긍정 문제는 잘 풀지만 부정 문제만 낮다면 산업 지식보다 `not`, `except`, `could not`, `all but one` 같은 표현을 놓치는 것이 원인일 수 있습니다. `None of the above` 유형만 낮다면 모델이 기존 선택지 중 하나를 억지로 고르는 경향이 있는지 확인해야 합니다.

### 3. 오답 분석표 만들기

모델이 틀린 문제마다 다음 정보를 기록합니다.

* 문제 ID
* 설비 종류(`asset_class`)
* 문제 유형(`family`)
* 핵심 센서 또는 고장(`anchor`)
* 실제 정답과 모델 예측
* 모델의 추론 내용
* 추정되는 오답 원인

오답 원인은 부정 표현 누락, `None of the above` 판단 실패, 산업 지식 부족, 선택지 문자 추출 오류 등으로 분류할 수 있습니다.

### 4. 프롬프트 개선하기

오답 분석 결과에 맞춰 프롬프트를 수정합니다. 다음과 같이 답을 고르기 전에 거쳐야 할 단계를 명시하는 방식이 유용합니다.

```text
1. Identify the asset class.
2. Determine whether the direction is failure-to-sensor or sensor-to-failure.
3. Check for negation such as not, except, could not, or all but one.
4. Evaluate every option using physical causality.
5. Check whether None of the above is appropriate.
6. Return one option letter and a concise rationale.
```

프롬프트를 변경할 때마다 동일한 검증 세트로 전체 정확도와 유형별 정확도를 다시 계산하여 이전 기준점과 비교합니다.

### 5. 모델 비교하기

동일한 Zero-shot 프롬프트를 여러 허용 모델에 적용해 비교합니다. 전체 정확도가 같더라도 부정 문제, `None of the above`, 특정 설비에서의 성능이 다를 수 있으므로 유형별 결과도 함께 비교합니다.

대회 규정에 맞게 오픈 소스 모델을 사용하고, 모델 크기 제한과 추론 환경 제한을 확인해야 합니다.

### 6. 필요한 경우 LoRA 파인튜닝하기

Zero-shot과 프롬프트 개선만으로 성능이 부족하고 대회 규정이 허용한다면, 검증 데이터 일부를 이용한 LoRA 학습을 고려합니다. 데이터가 크지 않으므로 산업 지식을 새로 학습시키기보다는 문제 형식, 부정 표현, 정답 출력 형식에 적응시키는 효과를 기대하는 편이 현실적입니다.

학습용과 평가용 데이터를 나눌 때 동일한 질문이 양쪽에 들어가면 정확도가 부풀려질 수 있습니다. 따라서 동일한 `question`은 같은 그룹에 넣고 분할해야 합니다. 더 엄격하게 평가하려면 다음 조합을 기준으로 그룹화할 수 있습니다.

```text
asset_class + family + anchor
```

### 7. 최종 Predictor 만들기

선택한 모델과 프롬프트를 `predict(scenario)` 함수에 연결합니다. 모델은 문제마다 다시 불러오지 말고 파일이 로드될 때 한 번만 초기화합니다.

```python
tokenizer = load_tokenizer()
model = load_model()


def predict(scenario):
    prompt = build_prompt(scenario)
    output = run_model(model, tokenizer, prompt)
    answer = extract_answer_letter(output, scenario.options)
    return {"answer": answer}
```

반환값은 반드시 `{"answer": "A"}`처럼 선택지 문자를 담은 딕셔너리여야 합니다. 모델이 설명까지 출력한다면 정답 문자만 안정적으로 추출하는 후처리가 필요합니다.

### 8. 테스트 예측 및 제출 검증하기

테스트 문제 전체에 대해 Predictor를 실행한 뒤 다음 항목을 확인합니다.

* 모든 문제 ID에 예측이 존재하는가?
* 반환된 답이 해당 문제의 유효한 선택지 문자인가?
* 누락되거나 중복된 문제가 없는가?
* 추론 코드가 인터넷, 검색 시스템 또는 외부 도구를 사용하지 않는가?
* 제출 노트북, 모델 정보, 추론 기록이 최신 제출 형식을 만족하는가?

현재 데이터 폴더에는 샘플 제출 파일이 없으므로, 실제 제출 전 Kaggle에서 최신 제출 형식과 대회 규칙을 다시 확인해야 합니다.
