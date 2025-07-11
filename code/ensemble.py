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

# main 함수 정의
def main():
    # argparse로 하이퍼파라미터 정의
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--img_size', type=int, default=300)
    parser.add_argument('--exp_name', type=str, default='ensemble_b4_swin')
    parser.add_argument('--data_dir', type=str, default='../input/data')
    args = parser.parse_args()

    # WandB 프로젝트 초기화
    wandb.init(project="cnn-doc-classification", name=args.exp_name)
    wandb.config.update(args)

    # 시드 고정
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 테스트 데이터 변환 정의
    tst_transform = A.Compose([
        A.LongestMaxSize(max_size=args.img_size),
        A.PadIfNeeded(min_height=args.img_size, min_width=args.img_size, border_mode=0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # 테스트 데이터셋 준비
    test_csv_path = os.path.join(args.data_dir, 'sample_submission.csv')
    test_img_dir = os.path.join(args.data_dir, 'test')
    tst_dataset = ImageDataset(test_csv_path, test_img_dir, transform=tst_transform)
    tst_loader = DataLoader(tst_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 앙상블할 모델 리스트 및 가중치 경로 설정
    model_names = ['efficientnet_b4', 'swin_tiny_patch4_window7_224']
    weights = ['efficientnet_b4_model.pth', 'swin_tiny_patch4_window7_224_model.pth']

    # 모델 불러오기
    models = []
    for name, weight in zip(model_names, weights):
        model = timm.create_model(name, pretrained=False, num_classes=17).to(device)
        model.load_state_dict(torch.load(weight, map_location=device))
        model.eval()
        models.append(model)

    # 앙상블 추론 수행
    preds_list = []
    for image, _ in tqdm(tst_loader):
        image = image.to(device)
        with torch.no_grad():
            preds = [model(image).softmax(dim=1) for model in models]
            avg_preds = torch.stack(preds).mean(dim=0)
            preds_list.extend(avg_preds.argmax(dim=1).cpu().numpy())

    # 결과를 DataFrame으로 정리 및 저장
    pred_df = pd.DataFrame(tst_dataset.df, columns=['ID', 'target'])
    pred_df['target'] = preds_list
    sample_submission_df = pd.read_csv(test_csv_path)
    assert (sample_submission_df['ID'] == pred_df['ID']).all()

    # 예측 결과 저장 (한국 시간 기준)
    KST = timezone(timedelta(hours=9))
    timestamp = datetime.now(KST).strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), "output"))
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{timestamp}_{args.exp_name}.csv")
    pred_df.to_csv(filename, index=False)
    print(f"예측 결과 저장 완료: {filename}")

    # WandB에 결과 artifact 업로드
    artifact = wandb.Artifact(args.exp_name, type='predictions')
    artifact.add_file(filename)
    wandb.log_artifact(artifact)

# main 실행
if __name__ == '__main__':
    main()
