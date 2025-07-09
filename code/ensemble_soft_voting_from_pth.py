import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta, timezone
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from timm import create_model
import albumentations as A
from albumentations.pytorch import ToTensorV2
import argparse

class TestDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_names = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(image=np.array(img))['image']
        return img, self.img_names[idx]

def get_transform(img_size, model_type="cnn"):
    if model_type == "transformer":
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

def get_tta_transforms(img_size):
    return [
        A.Compose([
            A.HorizontalFlip(p=1.0),
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        A.Compose([
            A.VerticalFlip(p=1.0),
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        A.Compose([
            A.RandomBrightnessContrast(p=1.0),
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
    ]

def run_soft_voting_from_fixed_pths(args):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CLASSES = 17
    pth_paths = [f"{args.base_name}_fold{fold}_best.pth" for fold in range(5)]
    img_dir = os.path.join(args.data_dir, "test")
    img_names = sorted(os.listdir(img_dir))

    if args.use_tta:
        print("🧪 TTA 기반 추론 시작...")
        tta_transforms = get_tta_transforms(args.img_size)
    else:
        print("🧪 일반 추론 시작...")
        tst_transform = get_transform(args.img_size, args.model_type)
        dataset = TestDataset(img_dir, transform=tst_transform)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    df_probs = []
    for pth in pth_paths:
        model = create_model(
            args.base_name,
            pretrained=False,
            num_classes=NUM_CLASSES,
            img_size=(args.img_size, args.img_size) if args.force_model_img_size else None
        )
        model.load_state_dict(torch.load(pth, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()

        probs_all = []

        if args.use_tta:
            for name in tqdm(img_names, desc=f"TTA Infer {os.path.basename(pth)}"):
                img_path = os.path.join(img_dir, name)
                img_raw = np.array(Image.open(img_path).convert("RGB"))

                tta_probs = []
                for tf in tta_transforms:
                    img_aug = tf(image=img_raw)['image'].unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        pred = model(img_aug)
                        prob = torch.softmax(pred, dim=1).cpu().numpy()
                        tta_probs.append(prob)

                mean_prob = np.mean(tta_probs, axis=0)
                probs_all.append(mean_prob[0])
        else:
            with torch.no_grad():
                for images, _ in tqdm(loader, desc=f"Infer {os.path.basename(pth)}"):
                    images = images.to(DEVICE)
                    logits = model(images)
                    probs = torch.softmax(logits, dim=1)
                    probs_all.extend(probs.cpu().numpy())

        df_probs.append(np.array(probs_all))

    ensemble_probs = np.mean(df_probs, axis=0)
    ensemble_preds = np.argmax(ensemble_probs, axis=1)

    submission = pd.read_csv(os.path.join(args.data_dir, "sample_submission.csv"))
    submission['target'] = ensemble_preds

    KST = timezone(timedelta(hours=9))
    timestamp = datetime.now(KST).strftime("%Y%m%d_%H%M%S")
    tta_tag = "_tta" if args.use_tta else ""
    filename = f"{timestamp}_{args.base_name}_soft_voting{tta_tag}.csv"
    os.makedirs("../output", exist_ok=True)
    submission.to_csv(os.path.join("../output", filename), index=False)
    print(f"\n✅ 저장 완료: ../output/{filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_name', type=str, default='coat_lite_medium') # 또는 'convnext_base', 
    parser.add_argument('--img_size', type=int, default=384)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--force_model_img_size', action='store_true') # 모델이 img_size를 강제 학습했을 경우
    parser.add_argument('--model_type', type=str, default='transformer')  # cnn or transformer
    parser.add_argument('--use_tta', action='store_true')
    parser.add_argument('--data_dir', type=str, default='../input/data')
    args = parser.parse_args()

    run_soft_voting_from_fixed_pths(args)


