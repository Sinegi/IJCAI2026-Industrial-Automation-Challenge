# Industrial-Automation-Challenge


| 방법             |   val acc | test acc |
| -------------- | --------: | -------- |
| Base           |     22.54 |          |
| 단순 학습          |     24.64 |          |
| Augmentation   |     53.86 |          |
| **csreg**      | **57.09** |          |
| csreg+listwise |     71.01 | 0.69     |
|                |           |          |


## csreg+listwise
```
# 백업본, hdd3에 어댑터 없을때
cd /home/user1/SG/EQ

# test submission (tta=1, ~30~40분)
bash run.sh submit --adapter artifacts/csreg_listwise_qwen3_8b/final_adapter --tta 1

# val 평가 (tta=1)
bash run.sh eval --adapter artifacts/csreg_listwise_qwen3_8b/final_adapter --tta 1
```

| Family | count | Acc |
|---|---|---|
| negation_sensor_to_failure | 200 | 87.5% |
| negation_failure_to_sensor | 192 | 84.4% |
| positive_sensor_to_failure | 320 | 83.8% |
| positive_failure_to_sensor | 330 | 83.6% |
| positive_failure_to_sensor_nota | 104 | 0.96% |
| positive_sensor_to_failure_nota | 96 | 0.0% |
