## Team

| ![이정민](https://avatars.githubusercontent.com/u/122961094?v=4) | ![김태현](https://avatars.githubusercontent.com/u/7031901?v=4) | ![문진숙](https://avatars.githubusercontent.com/u/204665219?v=4) | ![강연경](https://avatars.githubusercontent.com/u/5043251?v=4) | ![진 정](https://avatars.githubusercontent.com/u/87558804?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [이정민](https://github.com/lIllIlIIIll)             |            [김태현](https://github.com/huefilm)             |            [문진숙](https://github.com/June3723)             |            [강연경](https://github.com/YeonkyungKang)             |            [진 정](https://github.com/wlswjd)             |
|                            팀장                             |                            담당 역할                             |                            담당 역할                             |                            담당 역할                             |                            담당 역할                             |

## 0. Overview
### Environment
- AMD Ryzen Threadripper 3960X 24-Core Processor
- NVIDIA GeForce RTX 3090
- CUDA Version 12.2

### Requirements
- albumentations==1.3.1
- numpy==1.26.0
- timm==0.9.12
- torch==2.1.0
- torchvision=0.16.0
- scikit-learn=1.3.2

## 1. Competiton Info

### Overview

- [AIstage](https://stages.ai/en/competitions/356/overview/description)
- 문서는 금융, 보험, 물류, 의료 등 도메인을 가리지 않고 많이 취급됩니다. 이 대회는 다양한 종류의 문서 이미지의 클래스를 예측합니다.
- 문서 타입 데이터셋을 이용해 이미지 분류를 모델을 구축합니다. 주어진 문서 이미지를 입력 받아 17개의 클래스 중 정답을 예측하게 됩니다.

### Timeline

- 6월 30일 (월) 10:00 ~ 7월 10일 (목) 19:00

## 2. Components

### Directory


### <실험전 내용>
- **WandB** 설정이 필요할 수 있음 (웹로그인 및 토큰 입력)
- **offline_augment.py**  : 실행하면 input/data/augmented/ 에 증강이미지가 생성됨 
- (선택) **train_non_doc_classifier.py** : 문서/비문서 관련 이진분류기 학습용 실행하면 binary_non_doc_classifier.pth 파일이 생성됨.
 



### <실험>
1️⃣ **train_main.py**  : 

  증강이미지(augmented)만을 사용할지 원본(train+augmented)과 같이 사용할지는 아래 코드에서 필요한 것을 사용(안사용하는것 주석)

```
# 🔁 Offline 증강
# combined_df = pd.concat([df, aug_df], ignore_index=True) # 1.원본과 증강 데이터 모두 사용
combined_df = aug_df # 2.증강 데이터만 사용
```
  만일 이진 문서분류(train_non_doc_classifier)를 사용하기 싫으면  train_main.py 에서 아래 해당코드를 주석처리
```
pred_df = apply_non_doc_classifier(pred_df, tst_loader, device, all_probs, args, model_name='convnext_base')
```

메인 실험 argparsre 를 이용해서 콘솔에서 옵션을 정해가며 실행할 수 있음 (예: --image_size 380)

```
python train_main.py \
  --model_name efficientnet_b4 \
  --img_size 380 \
  --batch_size 20 \
  --lr 1e-3 \
  --epochs 30 \
  --early_stop 5 \
  --model_type cnn \
  --exp_name efficient_b4_seperate_doc_v2
```
2️⃣ 실험이 끝나면 output/폴더에 날짜+실험exp내용+fold{n}.csv 파일이 폴드갯수만큼 생성됨.
3️⃣ **ensemble_hard_voting_from_pth.py**  : output/폴더에 생긴 5개의 csv 파일을 하드보팅함.(결과파일명의 일부를 코드안 EXPERIMENT_NAME에 설정)
```
# ✅ 실험 이름 변수로 정의
EXPERIMENT_NAME = "convnext_offaug3_cunfuse_fixtrain_v1"
```
3️⃣ **ensemble_soft_voting_from_pth.py**  : code/ 밑에 생긴 5개의 best pth파일을 소프트보팅함.(안에 코드에서 실험한 모델과 image size등 설정)

```
python ensemble_soft_voting_from_pth.py \
  --base_name coat_lite_medium \
  --img_size 384 \
  --batch_size 16 \
  --force_model_img_size \
  --model_type transformer \
  --use_tta
```
 `--use_tta` : tta 사용시
`--force_model_img_size` : 강제로 이미지 사이즈 조정했을 때
4️⃣  output/폴더에 결과 앙상블 파일 (**~ensemble.csv**) 파일이 생성되면 download 하여 submissoin에 올린다.

```
upstage_cv_project/
├── code/               
│   ├── **train_main.py**     # ConvNeXt 기반 메인 학습 (오프라인 증강 포함)  1️⃣
│   ├── offline_augment.py # 오프라인 증강을 해주는 파일 (사전에 실행, 코드중간의 for j in range(3): 의 숫자만큼 배수)
│   ├── train_non_doc_classifier.py  # 2/16 이진 분류기 학습용 (사전에 한번 실행 하면 binary_non_doc_classifier.pth생성됨)
│   ├── binary_non_doc_classifier.pth # 2/16 이진 분류기 학습결과
│   ├── convnext_base_fold0_best.pth #fold0~fold4  best 모델 가중치 파일 2️⃣(총 개씩 생성됨, 모델이 같을경우 덮어질수 있음)
│   ├── ensemble_hard_voting_from_csv.py     # Hard voting 앙상블3️⃣ (결과파일명의 일부를 코드안 EXPERIMENT_NAME에 설정 )
│   ├── ensemble_soft_voting_from_pth.py     # Soft voting 앙상블3️⃣ (코드하단 run_soft_voting_from_fixed_pths에 설정 )
│   ├── confusion_matrix_utils.py    #  Confusion matrix 시각화 유틸  (선택)
│   └── ...
│
├── input/
│   ├── data/
│   │   ├── train/                   # 학습 이미지 원본
│   │   ├── test/                    # 테스트 이미지
│   │   ├── train.csv                # 학습 메타 정보
│   │   ├── sample_submission.csv    # 제출 양식
│   │   └── augmented/             # 오프라인 증강 이미지 (원본과 분리)
│   └── ...
│
├── output/                          # 📦 예측 결과 및 시각화 출력 (파일명은 자동으로 날짜+모델명+exp_name이 붙음)
│   ├── 20250707_130305_convnext_offaug3_cunfuse_fixtrain_v1_fold0.csv #2️⃣
│   ├── 20250707_130305_convnext_offaug3_cunfuse_fixtrain_v1_fold1.csv #2️⃣
│   ├── 20250707_130305_convnext_offaug3_cunfuse_fixtrain_v1_fold2.csv #2️⃣
│   ├── 20250707_130305_convnext_offaug3_cunfuse_fixtrain_v1_fold3.csv #2️⃣
│   ├── 20250707_130305_convnext_offaug3_cunfuse_fixtrain_v1_fold4.csv #2️⃣
│   ├── ... (fold별 예측)
│   ├── **20250708_123554_convnext_base_manual_soft_ensemble.csv# 최종 앙상블 결과**(Soft voting결과)** 4️⃣
│   └── **20250707_130305_convnext_offaug3_cunfuse_fixtrain_v1_ensemble.csv   # 최종 앙상블 결과**(Hard voting결과) 4️⃣
│
├── wandb/                    # wandb 관련
│
├── docs/ (선택)                  #  문서 (현재 내용없음)
│
├── README.md                        # 프로젝트 설명
└── requirements.txt                 # 패키지 리스트
```



## 3. Data descrption

### Dataset overview

- 1570장의 학습 이미지를 통해 3140장의 평가 이미지를 예측
- 데이터가 어떤 class를 가지고 있는지 설명하는 meta.csv와 각 이미지 파일과 label을 매치한 train.csv 제공
(0 계좌번호, 1 임신 의료비 지급 신청서, 2 자동차 계기판, 3 입·퇴원 확인서, 4 진단서, 5 운전면허증, 6 진료비 영수증, 7 외래 진료 증명서, 8 국민 신분증, 9 여권, 10 지불 확인서, 11 의약품 영수증, 12 처방전, 13 이력서, 14 의견 진술, 15 자동차 등록증, 16 자동차 등록판)
- car_dashboard와 vehicle_registration_plate 와 같은 주제와 상이한 이미지가 다수 있었고, flip, rotate, mixup 등이 되어 있는 이미지도 존재함


### EDA

- 클래스별 샘플 수가 불균형하여, 모델이 특정 클래스에 과적합될 가능성이 존재
- confusion matrix를 통해 어떤 클래스가 혼동되는지 확인하고, 해당 클래스를 중심으로 추가적인 피쳐 엔지니어링과 loss 조정 전략을 설계
- 일부 이미지의 라벨이 명확하지 않거나 불분명하여, 수동으로 확인 및 재정제 작업을 수행
- test 데이터는 여러 augmentation(회전, 뒤집기, mixup 등)이 적용되어 있었음

### Data Processing

- train.csv와 meta.csv를 기반으로 클래스와 이미지 간 매칭을 진행하였으며, 라벨이 잘못되거나 애매한 경우 직접 수정(수작업 제거함)
- 이미지 전처리에는 albumentations 라이브러리를 사용하여 회전, 노이즈 추가, 왜곡, 정규화 등의 변형을 적용
- 모델 성능 향상을 위해 17종의 offline augmentation을 생성하여 데이터셋을 확장하고, 학습 시 다양한 분포의 데이터를 노출시켜 모델의 일반화 성능을 높임
- 클래스 불균형 문제를 완화하기 위해 oversampling 기법을 일부 클래스에 적용하였고, focal loss의 알파 값을 클래스 비율에 따라 조정하는 방식으로 보정

## 4. Modeling

### Model descrition

- 주요 베이스라인 모델로 EfficientNet 계열을 선택(보편적으로 높은 성능과 효율성을 보여주었으며, 사전 학습된 모델이 제공되어서..)
- 추가로 ConvNeXt, CoAtNet, HRNet, coat_lite 등 다양한 모델 아키텍처를 실험하여, 클래스 간 성능 차이를 비교하고자 함
- EfficientNet B4 모델과 coat_lite_medium은 성능과 파라미터 수의 균형이 좋아 최종 모델 구조에 선택되었으며
- 다양한 CNN 모델과 더불어 K-FOld, 앙상블(Soft-Voting & Hard-Voting & Stacking) 기법을 다양하게 활용하여 성능을 향상시킴

### Modeling Process

- 모델은 timm 라이브러리를 활용하여 사전학습된 구조를 불러와 Transfer Learning 또는 Fine-Tuning 방식으로 학습을 진행
<pre><code>```
  model = timm.create_model(
    model_name,
    pretrained=True,
    num_classes=17
).to(device)
```</code></pre>
- Optimizer는 Adam을 사용하고, 학습률 스케줄러로는 LambdaLR 또는 CosineAnnealingWarmRestarts를 상황에 따라 적용
- 클래스 불균형을 고려하여 Focal Loss를 사용했으며, 클래스별 데이터 수에 따라 alpha 값을 조정해 손실을 계산(일부 실험에서는 classifier(head)만 학습한 후 전체 모델을 fine-tune해봤음)


## 5. Result

### Leader Board

<img width="1055" height="706" alt="image" src="https://github.com/user-attachments/assets/213ae09e-4579-4a6b-bda4-c3e74555814b" /># Title (Please modify the title)

- F1 Score : 0.9522

### Presentation

- [_13기 8조 presentaion file(pdf) link_](https://docs.google.com/presentation/d/16QH-98pLcHNOiJ13gGYMyGsNaQXYtBhh/edit?slide=id.p1#slide=id.p1)
