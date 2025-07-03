import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
# train_fold_kfold_with_infer.py
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
from sklearn.model_selection import StratifiedKFold
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

# Validation 평가 함수 정의
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

# 1 epoch 학습 함수 정의

# CutMix & MixUp utils
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def train_one_epoch(loader, model, optimizer, loss_fn, device, use_cutmix=False, use_mixup=False, alpha=1.0):
    model.train()
    train_loss = 0
    preds_list = []
    targets_list = []

    pbar = tqdm(loader)
    for image, targets in pbar:
        image = image.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)

        if use_cutmix and np.random.rand() < 0.5:
            lam = np.random.beta(alpha, alpha)
            rand_index = torch.randperm(image.size()[0]).to(device)
            target_a = targets
            target_b = targets[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(image.size(), lam)
            image[:, :, bbx1:bbx2, bby1:bby2] = image[rand_index, :, bbx1:bbx2, bby1:bby2]
            preds = model(image)
            loss = lam * loss_fn(preds, target_a) + (1 - lam) * loss_fn(preds, target_b)
        elif use_mixup and np.random.rand() < 0.5:
            lam = np.random.beta(alpha, alpha)
            rand_index = torch.randperm(image.size()[0]).to(device)
            mixed_image = lam * image + (1 - lam) * image[rand_index, :]
            target_a = targets
            target_b = targets[rand_index]
            preds = model(mixed_image)
            loss = lam * loss_fn(preds, target_a) + (1 - lam) * loss_fn(preds, target_b)
        else:
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

# Validation 평가 함수 정의
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
    parser.add_argument('--early_stop', type=int, default=5)
    parser.add_argument('--cutmix', action='store_true', help='Use CutMix augmentation')
    parser.add_argument('--mixup', action='store_true', help='Use MixUp augmentation')
    parser.add_argument('--mix_alpha', type=float, default=1.0, help='Alpha for CutMix/MixUp')

    args = parser.parse_args()

    # output 디렉토리 전역 생성 (confusion matrix, confused images, 예측 결과 모두 사용)
    output_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), "output"))
    os.makedirs(output_dir, exist_ok=True)

    # 클래스 이름 매핑 (meta.csv 활용)
    meta_path = os.path.join(args.data_dir, 'meta.csv')
    if os.path.exists(meta_path):
        meta_df = pd.read_csv(meta_path)
        class_names = meta_df['class_name'].tolist()
    else:
        class_names = [str(i) for i in range(17)]

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

    # Stratified K-Fold 기반 train/validation 분할
    train_csv_path = os.path.join(args.data_dir, 'train.csv')
    train_img_dir = os.path.join(args.data_dir, 'train')
    df = pd.read_csv(train_csv_path)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['target'])):
        print(f"\n🚀 Fold {fold} start")

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

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

        model = timm.create_model(args.model_name, pretrained=True, num_classes=17).to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_path = f"{args.model_name}_fold{fold}_best.pth"

        for epoch in range(args.epochs):
            metrics = train_one_epoch(
                trn_loader, model, optimizer, loss_fn, device,
                use_cutmix=args.cutmix, use_mixup=args.mixup, alpha=args.mix_alpha
            )
            val_metrics = evaluate(val_loader, model, loss_fn, device)
            metrics.update(val_metrics)
            metrics['epoch'] = epoch
            print(f"[Epoch {epoch}] Loss: {metrics['train_loss']:.4f}, Acc: {metrics['train_acc']:.4f}, F1: {metrics['train_f1']:.4f} | Val_Loss: {metrics['val_loss']:.4f}, Val_Acc: {metrics['val_acc']:.4f}, Val_F1: {metrics['val_f1']:.4f}")
            wandb.log(metrics)

            scheduler.step(metrics['val_loss'])

            if metrics['val_loss'] < best_val_loss:
                best_val_loss = metrics['val_loss']
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
                print(f"\n✅ New best model saved as {best_model_path} at epoch {epoch} with Val_Loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                print(f"\n⚠️ No improvement. patience_counter = {patience_counter}/{args.early_stop}")
                if patience_counter >= args.early_stop:
                    print(f"\n🛑 Early stopping triggered at epoch {epoch}.")
                    break

        # --- Confusion Matrix & 헷갈림 분석 ---
        print(f"\n[Fold {fold}] Confusion Matrix & Class Confusion Analysis")
        # Validation set 전체 예측값 수집
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for image, targets in val_loader:
                image = image.to(device)
                preds = model(image)
                val_preds.extend(preds.argmax(dim=1).detach().cpu().numpy())
                val_targets.extend(targets.detach().cpu().numpy())
        cm = confusion_matrix(val_targets, val_preds, labels=list(range(len(class_names))))
        plt.figure(figsize=(12,10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix (Fold {fold})')
        plt.tight_layout()
        cm_filename = os.path.join(output_dir, f"confusion_matrix_fold{fold}.png")
        plt.savefig(cm_filename)
        print(f"Confusion matrix saved: {cm_filename}")

        # 헷갈리는 쌍 상위 5개 출력
        cm_offdiag = cm.copy()
        np.fill_diagonal(cm_offdiag, 0)
        confusion_pairs = []
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                if i != j and cm_offdiag[i, j] > 0:
                    confusion_pairs.append((cm_offdiag[i, j], class_names[i], class_names[j]))
        confusion_pairs.sort(reverse=True)
        print("Top 5 most confused class pairs (val set):")
        for n, (cnt, true_cls, pred_cls) in enumerate(confusion_pairs[:5]):
            print(f"{n+1}. '{true_cls}' → '{pred_cls}' : {cnt}회")
        print("\n이 쌍들에 대해 augmentation 강화, loss tuning, 샘플 증강 등을 집중적으로 시도해보세요!")

        # --- 혼동된 validation 이미지 저장 ---
        confused_dir = os.path.join(output_dir, f"confused_val_imgs_fold{fold}")
        os.makedirs(confused_dir, exist_ok=True)
        val_df = pd.read_csv(val_df_path)
        saved = 0
        for idx, (true, pred) in enumerate(zip(val_targets, val_preds)):
            for _, true_cls, pred_cls in confusion_pairs[:5]:
                if class_names[true] == true_cls and class_names[pred] == pred_cls:
                    img_name = val_df.iloc[idx]['ID']
                    img_path = os.path.join(train_img_dir, img_name)
                    try:
                        img = Image.open(img_path)
                        img.save(os.path.join(confused_dir, f"{idx}_{true_cls}_to_{pred_cls}_{img_name}"))
                        saved += 1
                    except Exception as e:
                        print(f"이미지 저장 실패: {img_path}, 에러: {e}")
        print(f"혼동된 validation 이미지 {saved}개를 {confused_dir}에 저장했습니다.")

        # 추론 수행
        print(f"\n🧪 Inference using {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.eval()

        preds_list = []
        for image, _ in tqdm(tst_loader):
            image = image.to(device)
            with torch.no_grad():
                preds = model(image)
            preds_list.extend(preds.argmax(dim=1).detach().cpu().numpy())

        pred_df = pd.DataFrame(tst_dataset.df, columns=['ID', 'target'])
        pred_df['target'] = preds_list
        sample_submission_df = pd.read_csv(os.path.join(args.data_dir, 'sample_submission.csv'))
        assert (sample_submission_df['ID'] == pred_df['ID']).all()

        # 예측 결과 저장
        KST = timezone(timedelta(hours=9))
        timestamp = datetime.now(KST).strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), "output"))
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f"{timestamp}_{args.model_name}_{args.exp_name}_fold{fold}.csv")
        pred_df.to_csv(filename, index=False)
        print(f"📦 Fold {fold} 예측 결과 저장 완료: {filename}")

        artifact = wandb.Artifact(f"{args.exp_name}_fold{fold}", type='predictions')
        artifact.add_file(filename)
        wandb.log_artifact(artifact)

if __name__ == '__main__':
    main()
