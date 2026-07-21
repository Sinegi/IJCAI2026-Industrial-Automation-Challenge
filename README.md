# Industrial-Automation-Challenge


| 방법                                      |   val acc | test acc (제출) |
| --------------------------------------- | --------: | ------------- |
| Base                                    |     22.54 |               |
| 단순 학습                                   |     24.64 |               |
| Augmentation                            |     53.86 |               |
| **csreg**                               | **57.09** |               |
| csreg+listwise                          |     71.01 | 0.69          |
| +Threshold/abstain logit (NOTA (C+게이트)) |     77.13 | 0.67          |
| NOTA+A+B                                |     80.27 |               |


## csreg+listwise
```bash
# 백업본, hdd3에 어댑터 없을때 다시 학습해야함
cd /home/user1/SG/EQ

# test submission (tta=1, ~30~40분)
bash run.sh submit --tta 1

# val 평가 (tta=1)
bash run.sh eval --tta 1
```

| Family | count | Acc |
|---|---|---|
| negation_sensor_to_failure | 200 | 87.5% |
| negation_failure_to_sensor | 192 | 84.4% |
| positive_sensor_to_failure | 320 | 83.8% |
| positive_failure_to_sensor | 330 | 83.6% |
| positive_failure_to_sensor_nota | 104 | 0.96% |
| positive_sensor_to_failure_nota | 96 | 0.0% |
## +Threshold/abstain logit (NOTA 해결)
```bash
bash run.sh train
bash run.sh train --overwrite-output-dir # 재실행 시 기존 폴더를 지우고 새로 시작(첫 실행이면 없어도 무방)

bash run.sh eval --tta 1 --output-dir outputs/validation_nota

# 결과 확인
cat outputs/validation_nota/metrics.json
column -s, -t outputs/validation_nota/metrics_by_family.csv
```

## 📊 비교 종합 (val 1242, tta=1)

| 지표           | listwise  | **NOTA(이번)** | Δ           |
| ------------ | --------- | ------------ | ----------- |
| **전체 정확도**   | 71.01%    | **77.13%**   | **+6.1pp**  |
| 95% CI       | 68.5–73.6 | 75.0–79.3    | **비겹침(유의)** |
| macro_family | 56.70%    | **80.07%**   | **+23.4pp** |
| macro_asset  | 70.57%    | 76.17%       | +5.6pp      |

### Family별

|family|listwise|NOTA|Δ|
|---|---|---|---|
|positive_sensor_to_failure_**nota**|0.0%|**93.8%**|**+93.8**|
|positive_failure_to_sensor_**nota**|0.96%|**89.4%**|**+88.5**|
|negation_sensor_to_failure|87.5%|73.5%|−14.0|
|negation_failure_to_sensor|84.4%|74.5%|−9.9|
|positive_failure_to_sensor|83.6%|74.2%|−9.4|
|positive_sensor_to_failure|83.75%|75.0%|−8.75|

### 비-NOTA 정확도 (옵션 수별) — **균일 −10pp**

| n_opt         | listwise  | NOTA      | Δ         |
| ------------- | --------- | --------- | --------- |
| 4             | 92.9%     | 80.4%     | −12.5     |
| 5             | 89.9%     | 79.8%     | −10.1     |
| 6             | 84.4%     | 73.9%     | −10.5     |
| 7             | 81.9%     | 71.3%     | −10.6     |
| 8             | 75.2%     | 68.0%     | −7.2      |
| **비-NOTA 전체** | **84.5%** | **74.4%** | **−10.1** |

## 🔬 진단

**1. NOTA는 완전히 해결됐습니다.** 0~1% → **89~94%**. 예측 단위로 200문항 중 **183개에서 NOTA 슬롯을 정확히 선택**(이전 6개). abstain 로짓 + 합성행 설계가 정확히 의도대로 작동.

**2. 그러나 관계형(비-NOTA)이 균일하게 −10pp 후퇴.** 중요한 건 이게 **false-abstention 누수가 아니라는 것**:

- 비-NOTA 문항엔 애초에 NOTA 옵션이 없어 "잘못 abstain" 불가
- 예측이 마지막 슬롯에 쏠리지 않음(30% vs gold 34%), invalid 예측 0
- 오답이 전 위치에 고르게 분산, 옵션 수 전 구간에서 **일정하게** 하락

→ 즉 원인은 **표현공간의 전역적 "결단력 약화"**. NOTA 행에서 `gold=NOTA`인 softmax CE가 **모든 실제 옵션의 로짓을 끌어내리며**(→ 공유 LoRA 표현이 "옵션들은 anchor와 덜 관련" 쪽으로 이동), 여기에 NOTA 합성행의 "의심" prior가 겹쳐 관계형 랭킹의 예리함이 무뎌졌습니다.

**3. 순득실**: NOTA +182문항 / 관계형 −105문항 = 순 +77 → 71→77%. **명백한 개선이자 균형(macro_family +23pp)**. 하지만 관계형 10pp를 테이블에 남겨둔 상태.

## 🎯 개선 방향 (우선순위)

**A. abstain gradient를 관계형 표현에서 격리** ⭐ (가장 싸고 직접적) NOTA 행에서 실제 옵션 점수를 `.detach()`하여, CE가 **abstain 로짓만 끌어올리고 실제 옵션 표현은 끌어내리지 않게** 함. 진단 1의 근본 경로를 정확히 차단. `listwise_loss` 몇 줄 수정.

**B. "NOTA 옵션 있지만 정답은 실제 옵션"인 보정행 추가** 현재 NOTA 옵션이 있는 훈련행은 **전부 정답=NOTA** → 모델이 "NOTA 옵션 존재 → NOTA 정답" 편향 학습. edge가 정답이면서 NOTA 옵션도 포함한 행을 섞어 **answerable/unanswerable 경계 보정**(과잉 abstention 방지).

**C. NOTA 투여량/가중치 튜닝**: `nota_per_anchor`↓ 또는 NOTA 행 손실에 별도 작은 가중치. trade-off 곡선의 sweet spot 탐색.

**D. eval tta=3**: 순열 다수결로 관계형 안정화(이 모델은 순서에 답이 잘 바뀜). 무료로 관계형 1~3pp 회복 기대, NOTA엔 무해.

**E. 완전 분리(2번 답안)**: answerability를 별도 head로 빼 관계형 랭킹 손실을 오염 없이 유지. A/B가 부족하면 승급.

---
## + 보정 NOTA+A+B
```bash
bash run.sh train --overwrite-output-dir          # A+B 반영 재학습
bash run.sh eval --tta 1 --output-dir outputs/validation_nota_AB
```

### Family: A+B가 의도대로 트레이드오프를 되돌림

| family                          | NOTA  | **A+B**   | Δ     |
| ------------------------------- | ----- | --------- | ----- |
| positive_sensor_to_failure      | 75.0% | **84.4%** | +9.4  |
| negation_failure_to_sensor      | 74.5% | **82.8%** | +8.3  |
| negation_sensor_to_failure      | 73.5% | **81.5%** | +8.0  |
| positive_failure_to_sensor      | 74.2% | **80.6%** | +6.4  |
| positive_failure_to_sensor_nota | 89.4% | 70.2%     | −19.2 |
| positive_sensor_to_failure_nota | 93.8% | 68.8%     | −25.0 |

**관계형 74.4%→~82% 회복**(A 격리 + B 보정 효과 확인), NOTA는 90%→~69%로 일부 반납했지만 **여전히 baseline(0~5%) 대비 압도적**. 순효과 **+3.1pp**로 전체 최고.