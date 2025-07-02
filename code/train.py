#!/usr/bin/env python
# coding: utf-8

# # **📄 Document type classification baseline code**
# > 문서 타입 분류 대회에 오신 여러분 환영합니다! 🎉     
# > 아래 baseline에서는 ResNet 모델을 로드하여, 모델을 학습 및 예측 파일 생성하는 프로세스에 대해 알아보겠습니다.
# 
# ## Contents
# - Prepare Environments
# - Import Library & Define Functions
# - Hyper-parameters
# - Load Data
# - Train Model
# - Inference & Save File
# 

# ## 1. Prepare Environments
# 
# * 데이터 로드를 위한 구글 드라이브를 마운트합니다.
# * 필요한 라이브러리를 설치합니다.

# In[1]:


# 구글 드라이브 마운트, Colab을 이용하지 않는다면 패스해도 됩니다.
# from google.colab import drive
# drive.mount('/gdrive', force_remount=True)
# drive.mount('/content/drive')


# In[2]:


# 구글 드라이브에 업로드된 대회 데이터를 압축 해제하고 로컬에 저장합니다.
# get_ipython().system('tar -xvf drive/MyDrive/datasets_fin.tar > /dev/null')


# In[1]:
# 필요한 라이브러리를 설치합니다.
# get_ipython().system('pip install timm')


# ## 2. Import Library & Define Functions
# * 학습 및 추론에 필요한 라이브러리를 로드합니다.
# * 학습 및 추론에 필요한 함수와 클래스를 정의합니다.

# In[2]:
import os
import time
import random

import timm
import torch
import albumentations as A
import pandas as pd
import numpy as np
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.optim import Adam
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score


# In[3]:
# 시드를 고정합니다.
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True


# In[4]:
# 데이터셋 클래스를 정의합니다.
class ImageDataset(Dataset):
    def __init__(self, csv, path, transform=None):
        self.df = pd.read_csv(csv).values
        self.path = path
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        name, target = self.df[idx]
        img = np.array(Image.open(os.path.join(self.path, name)))
        if self.transform:
            img = self.transform(image=img)['image']
        return img, target


# In[17]:
# one epoch 학습을 위한 함수입니다.
def train_one_epoch(loader, model, optimizer, loss_fn, device):
    model.train()
    train_loss = 0
    preds_list = []
    targets_list = []

    pbar = tqdm(loader)
    for image, targets in pbar:
        image = image.to(device)
        targets = targets.to(device)

        model.zero_grad(set_to_none=True)

        preds = model(image)
        loss = loss_fn(preds, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds_list.extend(preds.argmax(dim=1).detach().cpu().numpy())
        targets_list.extend(targets.detach().cpu().numpy())

        pbar.set_description(f"Loss: {loss.item():.4f}")

    train_loss /= len(loader)
    train_acc = accuracy_score(targets_list, preds_list)
    train_f1 = f1_score(targets_list, preds_list, average='macro')

    ret = {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "train_f1": train_f1,
    }

    return ret


# ## 3. Hyper-parameters
# * 학습 및 추론에 필요한 하이퍼파라미터들을 정의합니다.

# In[31]:
# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# data config
data_path = 'input/data/'

# model config
model_name = 'resnet34' # 'resnet50' 'efficientnet-b0', ...

# training config
img_size = 128
LR = 1e-3
EPOCHS = 1
BATCH_SIZE = 32
num_workers = 0


# ## 4. Load Data
# * 학습, 테스트 데이터셋과 로더를 정의합니다.

# In[32]:
# augmentation을 위한 transform 코드
trn_transform = A.Compose([
    # 이미지 크기 조정
    A.Resize(height=img_size, width=img_size),
    # images normalization
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # numpy 이미지나 PIL 이미지를 PyTorch 텐서로 변환
    ToTensorV2(),
])

# test image 변환을 위한 transform 코드
tst_transform = A.Compose([
    A.Resize(height=img_size, width=img_size),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])


# In[33]:
# Dataset 정의
trn_dataset = ImageDataset(
    "../input/data/train.csv",
    "../input/data/train/",
    transform=trn_transform
)
tst_dataset = ImageDataset(
    "../input/data/sample_submission.csv",
    "../input/data/test/",
    transform=tst_transform
)
print(len(trn_dataset), len(tst_dataset))


# In[34]:
# DataLoader 정의
trn_loader = DataLoader(
    trn_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=False
)
tst_loader = DataLoader(
    tst_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)



# ## 5. Train Model
# * 모델을 로드하고, 학습을 진행합니다.

# In[38]:
# load model
model = timm.create_model(
    model_name,
    pretrained=True,
    num_classes=17
).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=LR)


# In[40]:
for epoch in range(EPOCHS):
    ret = train_one_epoch(trn_loader, model, optimizer, loss_fn, device=device)
    ret['epoch'] = epoch

    log = ""
    for k, v in ret.items():
      log += f"{k}: {v:.4f}\n"
    print(log)


# 
# 1570 3140
# Loss: 1.9827: 100%|██████████| 50/50 [00:05<00:00,  8.34it/s]
# train_loss: 1.7178
# train_acc: 0.4847
# train_f1: 0.4290
# epoch: 0.0000
# 
# 

# # 6. Inference & Save File
# * 테스트 이미지에 대한 추론을 진행하고, 결과 파일을 저장합니다.

# In[41]:
preds_list = []

model.eval()
for image, _ in tqdm(tst_loader):
    image = image.to(device)

    with torch.no_grad():
        preds = model(image)
    preds_list.extend(preds.argmax(dim=1).detach().cpu().numpy())


# In[42]:
pred_df = pd.DataFrame(tst_dataset.df, columns=['ID', 'target'])
pred_df['target'] = preds_list


# In[45]: 샘플_서브미션 파일의ID와 예측 할 파일과 매칭시킴
sample_submission_df = pd.read_csv("../input/data/sample_submission.csv")
assert (sample_submission_df['ID'] == pred_df['ID']).all()

# In[46]: 예측 csv파일 생성
from datetime import datetime, timedelta, timezone
KST = timezone(timedelta(hours=9))
timestamp = datetime.now(KST).strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join("..", "output")
os.makedirs(output_dir, exist_ok=True)
filename = os.path.join(output_dir, f"{timestamp}_{model_name}.csv")

pred_df.to_csv(filename, index=False)
print(f"예측 결과 저장 완료: {filename}")
