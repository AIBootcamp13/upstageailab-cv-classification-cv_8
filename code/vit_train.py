# vit_train.py
import os
import random
import argparse
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
import wandb

# --- Dataset ---
class TrainDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(os.path.join(self.image_dir, row['ID'])).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, row['target']

# --- Seed ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- Training ---
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, preds, targets = 0.0, [], []
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, lbls)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds.extend(outputs.argmax(dim=1).detach().cpu().numpy())
        targets.extend(lbls.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)
    f1 = f1_score(targets, preds, average='macro')
    return avg_loss, f1

# --- Validation ---
def validate(model, loader, criterion, device):
    model.eval()
    running_loss, preds, targets = 0.0, [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, lbls)

            running_loss += loss.item() * imgs.size(0)
            preds.extend(outputs.argmax(dim=1).cpu().numpy())
            targets.extend(lbls.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)
    f1 = f1_score(targets, preds, average='macro')
    return avg_loss, f1

# --- Main ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../input/data')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--exp_name', type=str, default='vit_train')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--wandb', action='store_true', help='Use wandb logging')
    parser.add_argument('--early_stop', type=int, default=3, help='Early stopping patience')
    args = parser.parse_args()

    if args.wandb:
        wandb.init(project="cnn-doc-classification", name=args.exp_name)
        wandb.config.update(args)

    set_seed(args.seed)

    df = pd.read_csv(os.path.join(args.data_dir, 'train.csv'))
    image_dir = os.path.join(args.data_dir, 'train')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['target'])):
        print(f"Fold {fold} start")
        if args.wandb:
            wandb.run.name = f"{args.exp_name}_fold{fold}"

        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
        train_dataset = TrainDataset(train_df, image_dir, transform)
        val_dataset = TrainDataset(val_df, image_dir, transform)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=17).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        best_f1 = 0
        patience = 0

        for epoch in range(args.epochs):
            train_loss, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_f1 = validate(model, val_loader, criterion, device)
            scheduler.step()

            if args.wandb:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_f1": train_f1,
                    "val_loss": val_loss,
                    "val_f1": val_f1
                })

            if val_f1 > best_f1:
                best_f1 = val_f1
                patience = 0
                torch.save(model.state_dict(), f"vit_base_fold{fold}_best.pth")
                print(f"✅ Model saved at fold {fold}, epoch {epoch}, val_f1: {val_f1:.4f}")
            else:
                patience += 1
                print(f"⏳ EarlyStopping counter: {patience}/{args.early_stop}")
                if patience >= args.early_stop:
                    print("🛑 Early stopping triggered!")
                    break

if __name__ == '__main__':
    main()
