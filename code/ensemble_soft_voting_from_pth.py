# ensemble_soft_voting_from_pth.py
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

# ✅ TestDataset 정의
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
        return img

# ✅ transform 정의

def get_transform(img_size, model_type="cnn"):
    if model_type == "transformer":
        return A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:  # cnn
        return A.Compose([
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

# ✅ soft voting 함수
def run_soft_voting_from_fixed_pths(config):
    base_name = config['base_name']
    img_size = config['img_size']
    batch_size = config['batch_size']
    force_model_img_size = config.get('force_model_img_size', False)
    model_type = config.get('model_type', 'cnn')

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CLASSES = 17

    print("[INFO] 모델 접두사:", base_name)

    pth_paths = [f"{base_name}_fold{fold}_best.pth" for fold in range(5)]

    test_dataset = TestDataset("../input/data/test", transform=get_transform(img_size, model_type))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    df_probs = []
    for path in pth_paths:
        model = create_model(
            base_name,
            pretrained=False,
            num_classes=NUM_CLASSES,
            img_size=(img_size, img_size) if force_model_img_size else None
        )
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()

        fold_probs = []
        with torch.no_grad():
            for images in tqdm(test_loader, desc=f"Infer {os.path.basename(path)}"):
                images = images.to(DEVICE)
                logits = model(images)
                probs = torch.softmax(logits, dim=1)
                fold_probs.append(probs.cpu().numpy())

        fold_probs = np.concatenate(fold_probs, axis=0)
        df_probs.append(fold_probs)

    ensemble_probs = np.mean(df_probs, axis=0)
    ensemble_preds = np.argmax(ensemble_probs, axis=1)

    submission = pd.read_csv("../input/data/sample_submission.csv")
    submission['target'] = ensemble_preds

    KST = timezone(timedelta(hours=9))
    timestamp = datetime.now(KST).strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{base_name}_manual_soft_ensemble.csv"
    output_path = os.path.join("../output", filename)
    submission.to_csv(output_path, index=False)
    print(f"\n✅ 수동 soft voting 결과 저장 완료: {output_path}")


# ✅ 직접 실행할 때 아래 부분만 수정해서 사용
if __name__ == '__main__':
    run_soft_voting_from_fixed_pths({
        "base_name": "convnext_base",
        "img_size": 380,
        "batch_size": 16,
        "force_model_img_size": False, # 모델이 img_size를 강제 학습했을 경우
        "model_type": "cnn"  # 또는 'transformer'
    })
