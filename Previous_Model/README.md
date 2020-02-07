#### TO-DO

Tensorflow 2.0, Python 3.8 version에 맞게 수정

#### 현재까지 Training 결과

| No |  Training Acc | Training Loss | Max Val Acc(Epoch) | Test Acc | kernel_init | pre-activation |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.9883 | 0.1519 | 0.9058 (347) | 0.9034 | 'glorot_uniform' | No |
| 2 | 0.9879 | 0.1657 | 0.8978 (346) | 0.8877 | 'he_normal' | No |
| 3 | 0.9887 | 0.1653 | 0.9148 (383) | 0.9106 | 'glorot_uniform' | Yes |
| 4 | 0.9894 | 0.1789 | 0.9114 (391) | 0.9022 | 'he_normal' | Yes |
| 5 | 0.9998 | 0.1261 | 0.9542(191) | 0.9501 | 'glorot_uniform' | Yes |
| 6 | 0.9999 | 0.1214 | 0.9558(189) | 0.9507 | 'he_normal' | Yes |
 
 Training Data : 45K - Validation Data : 5K - Test Data : 10K
 
 No. 1 ~ 4 model (4월 28일까지의 Commit을 기준으로 Training한 모델)
 400 Epoch까지 Training (SGD를 사용했고 learning rate는 0.01로 고정함)
 
 No 1, 2 : ResNet-112 ( 2-layers - 3 * 2 * 18 + 2 + 2 )
 
 No 3, 4 : ResNet-166 ( bottleneck structure - 3 * 3 * 18 + 2 + 2 )

 No.5 ~ No.6은 WRN-28-10-B(3,3) model이고, learning rate은 처음에는 0.1이며 60 epoch, 120epoch, 160epoch마다 learning rate에 0.2를 곱하고 200 Epoch까지 Training. 

| No | ResNet-{} | # params in my model | # params in Paper | Pre-Activation | Zero Pad | Bottleneck |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 20 | 269,722 | 0.27M | F | T | F |
| 2 | 32 | 464,154 | 0.46M | F | T | F |
| 3 | 44 | 658,586 | 0.66M | F | T | F |
| 4 | 56 | 853,018 | 0.85M | F | T | F |
| 5 | 110 | 1,727,962 | 1.7M | F | T | F |
| 6 | 1202 | 19,421,274 | 19.4M | F | T | F |
| 7 | 164 | 1,703,258 | 1.7M | T | F | T |
| 8 | 1001 | 10,327,706 | 10.2M | T | F | T |
| 9 | WRN-28-10 | 36,479,194 | 36.5M | T | F | B(3, 3) |
| 10 | WRN-22-8 | 17,158,106 | 17.2M | T | F | B(3, 3) |
| 11 | WRN-40-2 | 2,243,546 | 2.2M | T | F | B(3, 3) |
| 12 | WRN-16-10 | 17,116,634 | 17.1M | T | F | B(3, 3) |


#### 저장되는 파일들 예시

# model_result.txt
![model_result](https://user-images.githubusercontent.com/37528988/56464933-c81b0d00-642e-11e9-9977-bf3638df58b8.png)

# in_top_k.txt
![in-top-k](https://user-images.githubusercontent.com/37528988/56464942-eda81680-642e-11e9-8eee-7842c4769aa0.png)

# ckpt-info.json
![ckpt-info](https://user-images.githubusercontent.com/37528988/56721187-55d46080-677f-11e9-86e9-be51669b04c5.png)

