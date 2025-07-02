import argparse
import os
import time
import random
from datetime import datetime, timedelta, timezone

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
import wandb

# 데이터셋 클래스 정의
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

# 1 epoch 학습 함수 정의
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

    return {"train_loss": train_loss, "train_acc": train_acc, "train_f1": train_f1}

# main 함수 정의
def main():
    # argparse로 하이퍼파라미터 정의
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--model_name', type=str, default='resnet34')
    parser.add_argument('--exp_name', type=str, default='baseline-exp')
    parser.add_argument('--data_dir', type=str, default='../input/data')
    args = parser.parse_args()

    # WandB 프로젝트 초기화
    wandb.init(project="cnn-doc-classification", name=args.exp_name)
    wandb.config.update(args)

    # 시드 고정
    SEED = 42
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 데이터 변환 정의
    trn_transform = A.Compose([
        A.Resize(height=args.img_size, width=args.img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    tst_transform = A.Compose([
        A.Resize(height=args.img_size, width=args.img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # 데이터셋 및 로더 정의
    trn_dataset = ImageDataset(os.path.join(args.data_dir, 'train.csv'), os.path.join(args.data_dir, 'train'), transform=trn_transform)
    tst_dataset = ImageDataset(os.path.join(args.data_dir, 'sample_submission.csv'), os.path.join(args.data_dir, 'test'), transform=tst_transform)

    trn_loader = DataLoader(trn_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    tst_loader = DataLoader(tst_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 모델 정의
    model = timm.create_model(args.model_name, pretrained=True, num_classes=17).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    # 학습 루프
    for epoch in range(args.epochs):
        metrics = train_one_epoch(trn_loader, model, optimizer, loss_fn, device)
        metrics['epoch'] = epoch
        print(f"[Epoch {epoch}] Loss: {metrics['train_loss']:.4f}, Acc: {metrics['train_acc']:.4f}, F1: {metrics['train_f1']:.4f}")
        wandb.log(metrics)

    # 추론
    preds_list = []
    model.eval()
    for image, _ in tqdm(tst_loader):
        image = image.to(device)
        with torch.no_grad():
            preds = model(image)
        preds_list.extend(preds.argmax(dim=1).detach().cpu().numpy())

    pred_df = pd.DataFrame(tst_dataset.df, columns=['ID', 'target'])
    pred_df['target'] = preds_list
    sample_submission_df = pd.read_csv(os.path.join(args.data_dir, 'sample_submission.csv'))
    assert (sample_submission_df['ID'] == pred_df['ID']).all()

    # 예측 결과 저장 (한국 시간 기준)
    KST = timezone(timedelta(hours=9))
    timestamp = datetime.now(KST).strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(args.data_dir), 'output')
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{timestamp}_{args.exp_name}.csv")
    pred_df.to_csv(filename, index=False)
    print(f"예측 결과 저장 완료: {filename}")


    # wandb.save() 대신 artifact로 파일 저장
    artifact = wandb.Artifact(args.exp_name, type='predictions')
    artifact.add_file(filename)
    wandb.log_artifact(artifact)


if __name__ == '__main__':
    main()
