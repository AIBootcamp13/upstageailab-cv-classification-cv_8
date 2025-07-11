# train_15class_doc_model.py
# 2-Stage 구조용 문서 전용 15-class 모델 학습 코드 (불균형 보정 포함)

import os
import torch
import pandas as pd
import numpy as np
import random
from PIL import Image
from timm import create_model
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from tqdm import tqdm

# ==============================
# 🔹 Dataset 정의
# ==============================
class DocDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, 'ID']
        label = self.df.loc[idx, 'target']
        img = Image.open(os.path.join(self.img_dir, img_name)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

# ==============================
# 🔹 학습 함수
# ==============================
def train_15class_model(args):
    # 시드 고정
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    # 데이터 로딩 및 2, 16 제외
    df = pd.read_csv(os.path.join(args.data_dir, "train_100.csv"))
    df = df[~df['target'].isin([2, 16])].reset_index(drop=True)

    # 클래스 재매핑 (0~14)
    unique_targets = sorted(df['target'].unique())
    target_map = {old: new for new, old in enumerate(unique_targets)}
    df['target'] = df['target'].map(target_map)

    print(f"📊 훈련 클래스 수: {len(target_map)}개 → {target_map}")

    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold, (tr_idx, val_idx) in enumerate(skf.split(df, df['target'])):
        print(f"\n🚀 Fold {fold} 시작")
        train_df = df.iloc[tr_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        train_set = DocDataset(train_df, os.path.join(args.data_dir, "train"), transform)
        val_set = DocDataset(val_df, os.path.join(args.data_dir, "train"), transform)

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

        model = create_model(args.model_name, pretrained=True, num_classes=15, image_size=(args.img_size, args.img_size)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = torch.nn.CrossEntropyLoss()

        best_f1 = -1.0
        patience_counter = 0

        for epoch in range(args.epochs):
            model.train()
            total_loss = 0
            for imgs, targets in train_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                optimizer.zero_grad()
                output = model(imgs)
                loss = criterion(output, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            model.eval()
            val_preds, val_targets = [], []
            with torch.no_grad():
                for imgs, targets in val_loader:
                    imgs = imgs.to(device)
                    output = model(imgs)
                    preds = torch.argmax(output, dim=1).cpu().numpy()
                    val_preds.extend(preds)
                    val_targets.extend(targets.numpy())

            val_f1 = f1_score(val_targets, val_preds, average='macro')
            print(f"[Fold {fold}] Epoch {epoch} | Loss: {total_loss/len(train_loader):.4f} | Val F1: {val_f1:.4f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                save_path = f"{args.model_name}_15class_fold{fold}.pth"
                torch.save(model.state_dict(), save_path)
                print(f"✅ Best model saved to {save_path}")
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"⏳ No improvement. patience_counter = {patience_counter}/{args.early_stop}")
                if patience_counter >= args.early_stop:
                    print(f"🛑 Early stopping at epoch {epoch}.")
                    break

# ==============================
# 🔹 argparse
# ==============================
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

    train_15class_model(args)
