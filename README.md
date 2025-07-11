# Title (Please modify the title)
## Team

| ![이정민](https://avatars.githubusercontent.com/u/156163982?v=4) | ![김태현](https://avatars.githubusercontent.com/u/156163982?v=4) | ![문진숙](https://avatars.githubusercontent.com/u/156163982?v=4) | ![강연경](https://avatars.githubusercontent.com/u/156163982?v=4) | ![진 정](https://avatars.githubusercontent.com/u/156163982?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [이정민](https://github.com/UpstageAILab)             |            [김태현](https://github.com/UpstageAILab)             |            [문진숙](https://github.com/UpstageAILab)             |            [강연경](https://github.com/UpstageAILab)             |            [진 정](https://github.com/UpstageAILab)             |
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

- _Describe your EDA process and step-by-step conclusion_

### Data Processing

- _Describe data processing process (e.g. Data Labeling, Data Cleaning..)_

## 4. Modeling

### Model descrition

- _Write model information and why your select this model_

### Modeling Process

- _Write model train and test process with capture_

## 5. Result

### Leader Board

- _Insert Leader Board Capture_
- _Write rank and score_

### Presentation

- _Insert your presentaion file(pdf) link_

## etc

### Meeting Log

- _Insert your meeting log link like Notion or Google Docs_

### Reference

- _Insert related reference_
