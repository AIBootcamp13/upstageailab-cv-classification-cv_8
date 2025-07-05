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
    A.Rotate(limit=30, p=1.0),
    A.HorizontalFlip(p=1.0),
    A.RandomBrightnessContrast(p=1.0),
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
