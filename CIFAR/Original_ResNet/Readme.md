# ResNet 실행 방법

`python ResNet.py [Arguments] ...`

**ex)** 

`python ResNet.py -n 100`

---

| No | ResNet-{} | # parameters in my model | # parameters in Paper |
| --- | --- | --- | --- |
| 1 | 20 | 269,722 | 0.27M |
| 2 | 32 | 464,154 | 0.46M |
| 3 | 44 | 658,586 | 0.66M |
| 4 | 56 | 853,018 | 0.85M |
| 5 | 110 | 1,727,962 | 1.7M |
| 6 | 1202 | 19,421,274 | 19.4M |

---

### Result

참고) Label smoothing 적용 (-l "True")

##### learning rate 변화

| Epoch | Learning Rate |
| --- | --- |
| 1 ~ 200 | 0.01 |
| 201 ~ 300 | 0.005 |
| 301 ~ 400 | 0.001 |

| Model | Min Training Loss | Min Val Loss | Max Val Acc | Test Acc |
| --- | --- | --- | --- | --- |
| ResNet-110 | 0.501 | 0.768 | 0.922 | 0.915 |

---

**생성되는 파일 예시**

## **training_result_file**

![](./Pics/training_result.png)

---

## **top_k_accuracy**

![](./Pics/top_k_result.png)

---

## **training_result_summary**

![](./Pics/result_file_summary.png)
