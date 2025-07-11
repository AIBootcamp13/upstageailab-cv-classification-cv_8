# train_binary_doc_vs_nondoc.py
# 문서(1) vs 비문서(0) 이진 분류기 학습 스크립트

import os
import torch
import pandas as pd
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2
import albumentations as A
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from timm import create_model
from tqdm import tqdm
import wandb

class BinaryDocDataset(Dataset):
    def __init__(self, df, train_dir, aug_dir=None, transform=None):
        self.df = df.reset_index(drop=True)
        self.train_dir = train_dir
        self.aug_dir = aug_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, 'ID']
        label = self.df.loc[idx, 'target']

        if str(img_name).startswith("aug_") and self.aug_dir is not None:
            img_path = os.path.join(self.aug_dir, img_name)
        else:
            img_path = os.path.join(self.train_dir, img_name)

        img = np.array(Image.open(img_path).convert("RGB"))
        if self.transform:
            img = self.transform(image=img)['image']
        return img, label

def train_binary_model(args):
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    df1 = pd.read_csv(os.path.join(args.data_dir, "train_100.csv"))
    df2 = pd.read_csv(os.path.join(args.data_dir, "augmented.csv"))
    df = pd.concat([df1, df2], ignore_index=True)

    df['target'] = df['target'].apply(lambda x: 0 if x in [2, 16] else 1)  # 0 = 비문서, 1 = 문서

    print(df['target'].value_counts())

    transform = A.Compose([
        A.Resize(height=args.img_size, width=args.img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.4),
        A.RandomRotate90(p=0.4),
        A.Rotate(limit=20, p=0.4),
        A.GaussNoise(var_limit=(10.0, 30.0), p=0.3),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold, (tr_idx, val_idx) in enumerate(skf.split(df, df['target'])):
        print(f"\n🚀 Fold {fold} 시작")

        wandb.init(
            project="binary-doc-vs-nondoc",
            name=f"{args.model_name}_doc_binary_fold{fold}",
            config=vars(args)
        )

        train_df = df.iloc[tr_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        train_set = BinaryDocDataset(train_df, os.path.join(args.data_dir, "train"), aug_dir=os.path.join(args.data_dir, "augmented"), transform=transform)
        val_set = BinaryDocDataset(val_df, os.path.join(args.data_dir, "train"), aug_dir=os.path.join(args.data_dir, "augmented"), transform=transform)

        model = create_model(
            args.model_name,
            pretrained=True,
            num_classes=2,
            img_size=(args.img_size, args.img_size)
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = torch.nn.CrossEntropyLoss()

        best_f1 = -1.0
        patience_counter = 0

        for epoch in range(args.epochs):
            model.train()
            total_loss = 0
            train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
            for imgs, targets in tqdm(train_loader, desc=f"[Fold {fold}] Epoch {epoch} - Training"):
                imgs, targets = imgs.to(device), targets.to(device)
                optimizer.zero_grad()
                output = model(imgs)
                loss = criterion(output, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            model.eval()
            val_preds, val_targets = [], []
            val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
            with torch.no_grad():
                for imgs, targets in tqdm(val_loader, desc=f"[Fold {fold}] Epoch {epoch} - Validating"):
                    imgs = imgs.to(device)
                    output = model(imgs)
                    preds = torch.argmax(output, dim=1).cpu().numpy()
                    val_preds.extend(preds)
                    val_targets.extend(targets.numpy())

            val_f1 = f1_score(val_targets, val_preds, average='macro')
            avg_train_loss = total_loss / len(train_set)
            print(f"[Fold {fold}] Epoch {epoch} | Loss: {avg_train_loss:.4f} | Val F1: {val_f1:.4f}")

            wandb.log({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_f1": val_f1,
            })

            if val_f1 > best_f1:
                best_f1 = val_f1
                save_path = f"{args.model_name}_doc_binary_fold{fold}.pth"
                torch.save(model.state_dict(), save_path)
                print(f"✅ Best model saved to {save_path}")
                wandb.log({"best_model_path": save_path})
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"⏳ No improvement. patience_counter = {patience_counter}/{args.early_stop}")
                if patience_counter >= args.early_stop:
                    print(f"🛑 Early stopping at epoch {epoch}.")
                    break

        wandb.finish()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../input/data')
    parser.add_argument('--model_name', type=str, default='coat_lite_medium')
    parser.add_argument('--img_size', type=int, default=384)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--early_stop', type=int, default=5)
    args = parser.parse_args()

    train_binary_model(args)
