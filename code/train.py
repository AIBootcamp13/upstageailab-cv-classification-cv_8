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
from sklearn.model_selection import StratifiedShuffleSplit
import wandb

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from warmup_scheduler import GradualWarmupScheduler  # pip install warmup-scheduler

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
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--img_size', type=int, default=300)
    parser.add_argument('--model_name', type=str, default='efficientnet_b3')
    parser.add_argument('--exp_name', type=str, default='baseline')
    parser.add_argument('--data_dir', type=str, default='../input/data')
    parser.add_argument('--model_type', type=str, default='cnn', choices=['cnn', 'transformer'])
    args = parser.parse_args()

    # WandB 프로젝트 초기화
    wandb.init(project="cnn-doc-classification", name=f"{args.model_name}_{args.exp_name}")
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
    if args.model_type == 'transformer':
        trn_transform = A.Compose([
            A.Resize(height=args.img_size, width=args.img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.7),
            A.RandomRotate90(p=1.0),
            A.Rotate(limit=30, p=0.6),
            A.CenterCrop(height=args.img_size, width=args.img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

        tst_transform = A.Compose([
            A.Resize(height=args.img_size, width=args.img_size),
            A.CenterCrop(height=args.img_size, width=args.img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        trn_transform = A.Compose([
            A.LongestMaxSize(max_size=args.img_size),
            A.PadIfNeeded(min_height=args.img_size, min_width=args.img_size, border_mode=0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.7),
            A.RandomRotate90(p=1.0),
            A.Rotate(limit=30, p=0.6),
            A.GaussNoise(var_limit=(20.0, 60.0), p=0.5),
            A.MotionBlur(blur_limit=5, p=0.4),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.4),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4),
            A.RandomResizedCrop(height=args.img_size, width=args.img_size, scale=(0.6, 1.0), ratio=(0.8, 1.2), p=0.4),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

        tst_transform = A.Compose([
            A.LongestMaxSize(max_size=args.img_size),
            A.PadIfNeeded(min_height=args.img_size, min_width=args.img_size, border_mode=0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])


    # Stratified Split을 이용한 train/validation 분할
    train_csv_path = os.path.join(args.data_dir, 'train.csv')
    train_img_dir = os.path.join(args.data_dir, 'train')
    df = pd.read_csv(train_csv_path)
  
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, val_idx in sss.split(df['ID'], df['target']):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

    # 임시 csv 파일로 저장 (메모리 내 Dataset도 가능하지만, 기존 코드와 통일)
    train_df_path = os.path.join(args.data_dir, 'train_split.csv')
    val_df_path = os.path.join(args.data_dir, 'val_split.csv')
    train_df.to_csv(train_df_path, index=False)
    val_df.to_csv(val_df_path, index=False)

    trn_dataset = ImageDataset(train_df_path, train_img_dir, transform=trn_transform)
    val_dataset = ImageDataset(val_df_path, train_img_dir, transform=tst_transform)
    tst_dataset = ImageDataset(os.path.join(args.data_dir, 'sample_submission.csv'), os.path.join(args.data_dir, 'test'), transform=tst_transform)

    trn_loader = DataLoader(trn_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    tst_loader = DataLoader(tst_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 모델 정의
    model = timm.create_model(args.model_name, pretrained=True, num_classes=17).to(device)
    loss_fn = nn.CrossEntropyLoss()

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    # Early Stopping 설정
    early_stopping_patience = 5
    best_val_loss = float('inf')
    patience_counter = 0


    # Validation 평가 함수
    def evaluate(loader, model, loss_fn, device):
        model.eval()
        val_loss = 0
        preds_list = []
        targets_list = []
        with torch.no_grad():
            for image, targets in loader:
                image = image.to(device)
                targets = targets.to(device)
                preds = model(image)
                loss = loss_fn(preds, targets)
                val_loss += loss.item()
                preds_list.extend(preds.argmax(dim=1).detach().cpu().numpy())
                targets_list.extend(targets.detach().cpu().numpy())
        val_loss /= len(loader)
        val_acc = accuracy_score(targets_list, preds_list)
        val_f1 = f1_score(targets_list, preds_list, average='macro')
        return {"val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1}

    # 학습 루프

    for epoch in range(args.epochs):
        metrics = train_one_epoch(trn_loader, model, optimizer, loss_fn, device)
        val_metrics = evaluate(val_loader, model, loss_fn, device)
        metrics.update(val_metrics)
        metrics['epoch'] = epoch
        print(f"[Epoch {epoch}] Loss: {metrics['train_loss']:.4f}, Acc: {metrics['train_acc']:.4f}, F1: {metrics['train_f1']:.4f} | Val_Loss: {metrics['val_loss']:.4f}, Val_Acc: {metrics['val_acc']:.4f}, Val_F1: {metrics['val_f1']:.4f}")
        wandb.log(metrics)

        # ReduceLROnPlateau 스케줄러 적용 (val_loss 기준)
        scheduler.step(metrics['val_loss'])

        # Early Stopping
        if metrics['val_loss'] < best_val_loss:
            best_val_loss = metrics['val_loss']
            patience_counter = 0
            model_save_path = f"{args.model_name}_model.pth"
            torch.save(model.state_dict(), model_save_path)
            print(f"\n✅ New best model saved as {model_save_path} at epoch {epoch} with Val_Loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"\n⚠️ No improvement. patience_counter = {patience_counter}/{early_stopping_patience}")
            if patience_counter >= early_stopping_patience:
                print(f"\n🛑 Early stopping triggered at epoch {epoch}.")
                break

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
    output_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), "output"))
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{timestamp}_{args.model_name}_{args.exp_name}.csv")
    pred_df.to_csv(filename, index=False)
    print(f"예측 결과 저장 완료: {filename}")


    # wandb.save() 대신 artifact로 파일 저장
    artifact = wandb.Artifact(args.exp_name, type='predictions')
    artifact.add_file(filename)
    wandb.log_artifact(artifact)


if __name__ == '__main__':
    main()
