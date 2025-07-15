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

- _Insert your directory structure_

e.g.
```
├── code
│   ├── jupyter_notebooks
│   │   └── model_train.ipynb
│   └── train.py
├── docs
│   ├── pdf
│   │   └── (Template) [패스트캠퍼스] Upstage AI Lab 1기_그룹 스터디 .pptx
│   └── paper
└── input
    └── data
        ├── eval
        └── train
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
- 모델 성능 향상을 위해 16종의 offline augmentation을 생성하여 데이터셋을 확장하고, 학습 시 다양한 분포의 데이터를 노출시켜 모델의 일반화 성능을 높임
- 클래스 불균형 문제를 완화하기 위해 oversampling 기법을 일부 클래스에 적용하였고, focal loss의 알파 값을 클래스 비율에 따라 조정하는 방식으로 보정

## 4. Modeling

### Model descrition

- 주요 베이스라인 모델로 EfficientNet 계열을 선택(보편적으로 높은 성능과 효율성을 보여주었으며, 사전 학습된 모델이 제공되어서..)
- 추가로 ConvNeXt, CoAtNet, HRNet 등 다양한 모델 아키텍처를 실험하여, 클래스 간 성능 차이를 비교하고자 함
- EfficientNet V2 M 모델은 성능과 파라미터 수의 균형이 좋아 최종 모델 구조에 선택되었으며, 이를 활용해 Two-Stage 모델 구조(자동차 관련 vs 문서 관련 이진 분류 → 세부 분류)를 구성해 성능을 향상시킴
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

- [_Insert your presentaion file(pdf) link_](https://docs.google.com/presentation/d/16QH-98pLcHNOiJ13gGYMyGsNaQXYtBhh/edit?slide=id.p1#slide=id.p1)
