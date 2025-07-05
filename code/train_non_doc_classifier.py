# train_non_doc_classifier.py
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
            img = self.transform(image=img)['image']  # ✅ Albumentations expects dict input

        label = 1 if row['target'] in NON_DOC_CLASSES else 0
        return img, label

def train_binary_classifier(args):
    df = pd.read_csv(os.path.join(args.data_dir, 'train.csv'))

    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df['target'], random_state=SEED
    )

    transform = A.Compose([
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

    train_dataset = BinaryDocDataset(train_df, os.path.join(args.data_dir, 'train'), transform)
    val_dataset = BinaryDocDataset(val_df, os.path.join(args.data_dir, 'train'), transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = timm.create_model('efficientnet_b2', pretrained=True, num_classes=2)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        pred_list, target_list = [], []

        for x, y in tqdm(train_loader):
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

        # validation
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

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), "binary_non_doc_classifier.pth")
            print(f"✅ Saved model at epoch {epoch}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../input/data')
    parser.add_argument('--img_size', type=int, default=380)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()
    train_binary_classifier(args)
