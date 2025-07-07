import os
import cv2
import pandas as pd
import albumentations as A
from tqdm import tqdm

# 경로 설정
input_dir = "../input/data/train"
output_dir = "../input/data/augmented"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv("../input/data/train.csv")
augmented_records = []

# 증강 파이프라인
transform = A.Compose([
    A.Rotate(limit=30, p=0.7, border_mode=cv2.BORDER_CONSTANT, value=(255,255,255)),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
    A.MotionBlur(blur_limit=5, p=0.3),
    A.ISONoise(p=0.2),
    A.Downscale(scale_min=0.7, scale_max=0.9, p=0.2),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
])

for i in tqdm(range(len(df))):
    row = df.iloc[i]
    img_id = row["ID"]
    label = row["target"]
    
    img_path = os.path.join(input_dir, img_id)
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for j in range(3):  # 각 이미지당 3개 버전
        augmented = transform(image=image)["image"]
        aug_id = f"aug_{i:04d}_{j}.jpg"
        save_path = os.path.join(output_dir, aug_id)
        cv2.imwrite(save_path, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))
        augmented_records.append([aug_id, label])

# 증강 CSV 저장
df_aug = pd.DataFrame(augmented_records, columns=["ID", "target"])
df_aug.to_csv("../input/data/augmented.csv", index=False)
print("✅ 증강 이미지 및 CSV 저장 완료!")
