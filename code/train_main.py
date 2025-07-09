# train_main.py
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
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
import wandb

from torch.optim.lr_scheduler import ReduceLROnPlateau
from warmup_scheduler import GradualWarmupScheduler

# 데이터셋 클래스 정의
class ImageDataset(Dataset):
    def __init__(self, df, train_dir, aug_dir=None, transform=None):
        self.df = df.values if isinstance(df, pd.DataFrame) else pd.read_csv(df).values
        self.train_dir = train_dir
        self.aug_dir = aug_dir
        self.transform = transform

    def __getitem__(self, idx):
        name, target = self.df[idx]
        # 증강 이미지인지 확인
        if str(name).startswith("aug_") and self.aug_dir is not None:
            img_path = os.path.join(self.aug_dir, name)
        else:
            img_path = os.path.join(self.train_dir, name)

        img = np.array(Image.open(img_path).convert("RGB"))
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, target

    def __len__(self):
        return len(self.df)

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
    return {
        "val_loss": val_loss,
        "val_acc": val_acc,
        "val_f1": val_f1,
        "val_preds": preds_list,
        "val_targets": targets_list
    }

# main 함수 정의
def main():
    # argparse로 하이퍼파라미터 정의
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--img_size', type=int, default=300)
    parser.add_argument('--model_name', type=str, default='coat_lite_medium')
    parser.add_argument('--exp_name', type=str, default='baseline')
    parser.add_argument('--data_dir', type=str, default='../input/data')
    parser.add_argument('--model_type', type=str, default='cnn', choices=['cnn', 'transformer'])
    parser.add_argument('--early_stop', type=int, default=5)
    parser.add_argument('--cutmix', action='store_true', help='Use CutMix augmentation')
    parser.add_argument('--mixup', action='store_true', help='Use MixUp augmentation')
    parser.add_argument('--mix_alpha', type=float, default=1.0, help='Alpha for CutMix/MixUp')
    parser.add_argument('--use_tta', action='store_true', help='Use TTA at inference')
    # parser.add_argument('--label_smoothing', type=float, default=0.0, help='Label smoothing factor (0.0 to disable)')
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

            # 📄 문서 구조 깨지지 않게 약한 augmentation 위주
            A.HorizontalFlip(p=0.4),           # 문서 좌우 반전: 약하게만
            A.VerticalFlip(p=0.2),             # 상하 반전은 더 약하게
            A.Rotate(limit=15, p=0.3),         # 약한 회전 (문서 틀어짐 대응)
            A.RandomRotate90(p=0.2),           # 비대칭 문서 대응용

            # 💡 색상/노이즈 조절 (Transformer는 global context 학습이므로 약하게)
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.2),
            A.GaussNoise(var_limit=(10.0, 30.0), p=0.2),

            # ✅ 가장 중요: 강제 Resize 후 Normalize
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2()
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

    # train_csv_path = os.path.join(args.data_dir, 'train.csv')
    train_csv_path = os.path.join(args.data_dir, 'train_100.csv') # 클래스별 100장씩 
    aug_csv_path = os.path.join(args.data_dir, 'augmented.csv')
    train_img_dir = os.path.join(args.data_dir, 'train')
    aug_img_dir = os.path.join(args.data_dir, 'augmented')

    df = pd.read_csv(train_csv_path)
    aug_df = pd.read_csv(aug_csv_path)

    # 🔁 Offline 증강
    combined_df = pd.concat([df, aug_df], ignore_index=True) # 1.원본과 증강 데이터 모두 사용
    # combined_df = aug_df # 2.증강 데이터만 사용


    # 2. K-Fold split
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(skf.split(combined_df, combined_df['target'])):
        print(f"\n🚀 Fold {fold} start")

        train_df = combined_df.iloc[train_idx].reset_index(drop=True)
        val_df = combined_df.iloc[val_idx].reset_index(drop=True)

        train_df_path = os.path.join(args.data_dir, 'train_split.csv')
        val_df_path = os.path.join(args.data_dir, 'val_split.csv')
        train_df.to_csv(train_df_path, index=False)
        val_df.to_csv(val_df_path, index=False)

        trn_dataset = ImageDataset(train_df, train_img_dir, aug_dir=aug_img_dir, transform=trn_transform)
        val_dataset = ImageDataset(val_df, train_img_dir, aug_dir=aug_img_dir, transform=tst_transform)
        # 테스트 데이터셋은 sample_submission.csv를 사용
        tst_dataset = ImageDataset(os.path.join(args.data_dir, 'sample_submission.csv'), os.path.join(args.data_dir, 'test'), transform=tst_transform)

        trn_loader = DataLoader(trn_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        tst_loader = DataLoader(tst_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        # EfficientNetV2-S: Hugging Face 경로로 자동 치환
        if args.model_name == "efficientnetv2_s":
            model_name = "efficientnetv2_rw_s"
        else:
            model_name = args.model_name

        model = timm.create_model(model_name, pretrained=True, num_classes=17, img_size=(384, 384)).to(device)

        # loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

        best_val_f1 = -1.0  # <-- epoch 루프 전에 초기화!
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
            # wandb.log() 직전 처리
            metrics.pop("val_preds", None)
            metrics.pop("val_targets", None)
            wandb.log(metrics)

            # ✅ Confusion Matrix 로

            wandb.log({
                "val_confusion_matrix": wandb.plot.confusion_matrix(
                    y_true=val_metrics["val_targets"],    # 정답 라벨
                    preds=val_metrics["val_preds"],       # 예측 라벨
                    class_names=[str(i) for i in range(17)]
                )
            })

            # ✅ 모델 저장 기준 및 스케줄러 기준도 val_f1 기준으로 수정
            scheduler.step(metrics['val_f1'])

            # F1 기준으로 best model 저장

            if metrics['val_f1'] > best_val_f1 + 1e-5:  # val_f1 기준으로 최고 성능 모델 저장
                best_val_f1 = metrics['val_f1']
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
                print(f"\n✅ New best model saved as {best_model_path} at epoch {epoch} with Val_F1: {best_val_f1:.4f}")
            else:
                patience_counter += 1
                print(f"\n⚠️ No improvement. patience_counter = {patience_counter}/{args.early_stop}")
                if patience_counter >= args.early_stop:
                    print(f"\n🛑 Early stopping triggered at epoch {epoch}.")
                    break

        # 추론 수행
        print(f"\n🧪 Inference using {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.eval()

        # ✅ Test-Time Augmentation 정의
        tta_transforms = [
            A.Compose([
                A.Resize(height=args.img_size, width=args.img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]),
            A.Compose([
                A.HorizontalFlip(p=1.0),
                A.Resize(height=args.img_size, width=args.img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]),
            A.Compose([
                A.VerticalFlip(p=1.0),
                A.Resize(height=args.img_size, width=args.img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]),
            A.Compose([
                A.Rotate(limit=30, p=1.0),
                A.Resize(height=args.img_size, width=args.img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]),
            A.Compose([
                A.GaussNoise(var_limit=(10.0, 40.0), p=1.0),
                A.Resize(height=args.img_size, width=args.img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]),
        ]

        # ✅ 추론
        all_probs = []
        preds_list = []

        if args.use_tta:
            print("🧪 TTA 적용 중...")

            img_dir = os.path.join(args.data_dir, "test")
            img_names = pd.read_csv(os.path.join(args.data_dir, 'sample_submission.csv'))['ID'].tolist()

            for name in tqdm(img_names):
                img_path = os.path.join(img_dir, name)
                img_raw = np.array(Image.open(img_path).convert("RGB"))

                tta_preds = []

                for tf in tta_transforms:
                    augmented = tf(image=img_raw)["image"]
                    augmented = augmented.unsqueeze(0).to(device)

                    with torch.no_grad():
                        pred = model(augmented)
                        prob = torch.softmax(pred, dim=1).cpu().numpy()
                        tta_preds.append(prob)

                avg_prob = np.mean(tta_preds, axis=0)
                all_probs.append(avg_prob[0])
                preds_list.append(np.argmax(avg_prob[0]))

        else:
            for image, _ in tqdm(tst_loader):
                image = image.to(device)
                with torch.no_grad():
                    preds = model(image)
                    probs = torch.softmax(preds, dim=1).cpu().numpy()
                    all_probs.extend(probs)
                    preds_list.extend(np.argmax(probs, axis=1))


        pred_df = pd.DataFrame(tst_dataset.df, columns=['ID', 'target'])
        pred_df['target'] = preds_list

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
