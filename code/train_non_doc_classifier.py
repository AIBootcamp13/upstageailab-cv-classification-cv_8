import os
import random
import argparse

import pandas as pd
import numpy as np
from PIL import Image
import timm
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

NON_DOC_CLASSES = [2, 16]  # ✅ 계기판, 번호판

class BinaryDocDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['ID'])
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)

        if self.transform:
            img = self.transform(image=img)['image']

        label = 1 if row['target'] in NON_DOC_CLASSES else 0
        return img, label

def train_binary_classifier(args):
    df = pd.read_csv(os.path.join(args.data_dir, 'train.csv'))

    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df['target'], random_state=SEED
    )

    transform = A.Compose([
        A.Resize(args.img_size, args.img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    train_dataset = BinaryDocDataset(train_df, os.path.join(args.data_dir, 'train'), transform)
    val_dataset = BinaryDocDataset(val_df, os.path.join(args.data_dir, 'train'), transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = timm.create_model(args.model_name, pretrained=True, num_classes=2)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_f1 = -1.0
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        pred_list, target_list = [], []

        for x, y in tqdm(train_loader, desc=f"[Train] Epoch {epoch}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pred_list.extend(pred.argmax(1).detach().cpu().numpy())
            target_list.extend(y.cpu().numpy())

        train_f1 = f1_score(target_list, pred_list)
        print(f"[Epoch {epoch}] Train Loss: {train_loss/len(train_loader):.4f}, Train F1: {train_f1:.4f}")

        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_preds.extend(pred.argmax(1).cpu().numpy())
                val_targets.extend(y.cpu().numpy())

        val_f1 = f1_score(val_targets, val_preds)
        val_acc = accuracy_score(val_targets, val_preds)
        print(f"💡 Val F1: {val_f1:.4f} | Val Acc: {val_acc:.4f}")

        if val_f1 > best_f1 + 1e-5:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), "binary_non_doc_classifier.pth")
            print(f"✅ Saved best model at epoch {epoch} with Val F1: {best_f1:.4f}")
        else:
            patience_counter += 1
            print(f"⚠️ No improvement. patience_counter = {patience_counter}/{args.early_stop}")
            if patience_counter >= args.early_stop:
                print(f"🛑 Early stopping triggered at epoch {epoch}")
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../input/data')
    parser.add_argument('--img_size', type=int, default=384)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--early_stop', type=int, default=5)
    parser.add_argument('--model_name', type=str, default='convnext_base')
    parser.add_argument('--model_type', type=str, default='cnn')  # reserved
    args = parser.parse_args()

    train_binary_classifier(args)
