from transformers import AutoModelForImageClassification, AutoProcessor, BeitImageProcessor, BeitForImageClassification
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
import albumentations as A
from albumentations.pytorch import ToTensorV2
        
import os
from tqdm import tqdm
import pandas as pd
from PIL import Image

from torchvision import transforms
import numpy as np

print(f"PyTorch version: {torch.__version__}")

# Initialize processor and model
# Beit-base-patch16-224 : https://huggingface.co/microsoft/beit-base-patch16-224
processor = BeitImageProcessor.from_pretrained("microsoft/beit-base-patch16-224")
model = BeitForImageClassification.from_pretrained(
    "microsoft/beit-base-patch16-224", 
    num_labels=17,
    ignore_mismatched_sizes=True  # This will ignore the classifier size mismatch
)

class DocumentAugmentation:
    def __init__(self):
        
        # Define augmentation pipeline with Albumentations
        self.train_transform = A.Compose([
            # 1. Rotation (90도 회전 포함 다양한 각도)
            A.Rotate(limit=90, p=0.8),
            
            # 2. Noise (다양한 노이즈 추가)
            A.OneOf([
                A.ISONoise(p=1.0),
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.MultiplicativeNoise(multiplier=(0.5, 1.5), p=1.0)
            ], p=0.7),
            
            # 3. Blur (흐릿한 이미지 시뮬레이션)
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MotionBlur(blur_limit=(3, 7), p=1.0)
            ], p=0.6),
            
            # 4. 대비/밝기 저하
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.3,
                    contrast_limit=0.3,
                    p=1.0
                ),
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=1.0
                )
            ], p=0.6),
            
            # 5. JPEG 압축 아티팩트
            A.ImageCompression(quality_lower=30, quality_upper=100, p=0.5),
            
            # 6. Perspective 왜곡
            A.Perspective(scale=(0.02, 0.08), p=0.5),
            
            # 7. Cutout/CoarseDropout (정보 손실 시뮬레이션)
            A.OneOf([
                A.Cutout(num_holes=8, max_h_size=32, max_w_size=32, p=1.0),
                A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=1.0)
            ], p=0.4),
        ])
    
    def __call__(self, image):
        # Convert PIL image to numpy array
        image_np = np.array(image)
        # Apply augmentations (without normalization)
        augmented = self.train_transform(image=image_np)
        # Convert back to PIL Image for processor
        return Image.fromarray(augmented['image'])


class ImageDataset(Dataset):
    def __init__(self, csv_path, img_path, processor, is_train=True):
        self.df = pd.read_csv(csv_path).values
        self.img_path = img_path
        self.processor = processor
        self.is_train = is_train
        if is_train:
            self.augmentation = DocumentAugmentation()
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        name, target = self.df[idx]
        img = Image.open(os.path.join(self.img_path, name)).convert('RGB')
        
        # Apply augmentation for training data
        if self.is_train:
        # 1. Albumentations: augmentation만 (normalization 없음)
            img = self.augmentation(img)
        
        # 2. BeitImageProcessor: 자체 normalization 수행
        inputs = self.processor(images=img, return_tensors="pt")
        # 내부적으로 normalization 적용됨
        
        # Remove batch dimension and add label
        pixel_values = inputs["pixel_values"].squeeze(0)
        
        return {"pixel_values": pixel_values, "labels": int(target)}

def train_model(model, train_dataloader, device, num_epochs=20, learning_rate=5e-5, save_path=None):
    """
    모델을 훈련하는 함수
    
    Args:
        model: 훈련할 모델
        train_dataloader: 훈련 데이터 로더
        device (torch.device): 사용할 디바이스
        num_epochs (int): 훈련 에포크 수
        learning_rate (float): 학습률
        save_path (str): 모델 저장 경로 (None이면 저장하지 않음)
    
    Returns:
        list: 각 에포크의 평균 손실 리스트
    """
    # Training settings
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    model.to(device)
    
    print(f"Starting training for {num_epochs} epochs...")
    print(f"Learning rate: {learning_rate}")
    print(f"Device: {device}")
    print(f"Dataset size: {len(train_dataloader.dataset)}")
    
    # Training loop
    model.train()
    epoch_losses = []
    
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        
        for batch in progress_bar:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        
        avg_loss = total_loss / len(train_dataloader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
    
    # Save model if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save_pretrained(save_path)
        print(f"Model saved to {save_path}")
    
    return epoch_losses

def create_training_dataset(csv_path, img_path, processor, batch_size=8):
    """
    훈련 데이터셋과 데이터로더를 생성하는 함수
    
    Args:
        csv_path (str): 훈련 CSV 파일 경로
        img_path (str): 훈련 이미지 폴더 경로
        processor: 이미지 프로세서
        batch_size (int): 배치 크기
    
    Returns:
        tuple: (dataset, dataloader)
    """
    print(f"Creating training dataset from {csv_path}")
    train_dataset = ImageDataset(csv_path, img_path, processor, is_train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Dataset created with {len(train_dataset)} samples")
    return train_dataset, train_dataloader

def save_model_and_processor(model, processor, model_save_path, processor_save_path):
    """
    모델과 프로세서를 저장하는 함수
    
    Args:
        model: 저장할 모델
        processor: 저장할 프로세서
        model_save_path (str): 모델 저장 경로
        processor_save_path (str): 프로세서 저장 경로
    """
    print("Saving model and processor...")
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(processor_save_path), exist_ok=True)
    
    # Save model and processor
    model.save_pretrained(model_save_path)
    processor.save_pretrained(processor_save_path)
    
    print(f"Model saved to: {model_save_path}")
    print(f"Processor saved to: {processor_save_path}")

def train_and_save_model(model, processor, train_csv_path, train_img_path, 
                        model_save_path, processor_save_path, 
                        num_epochs=20, learning_rate=5e-5, batch_size=8):
    """
    모델을 훈련하고 저장하는 통합 함수
    
    Args:
        model: 훈련할 모델
        processor: 이미지 프로세서
        train_csv_path (str): 훈련 CSV 파일 경로
        train_img_path (str): 훈련 이미지 폴더 경로
        model_save_path (str): 모델 저장 경로
        processor_save_path (str): 프로세서 저장 경로
        num_epochs (int): 훈련 에포크 수
        learning_rate (float): 학습률
        batch_size (int): 배치 크기
    
    Returns:
        list: 훈련 손실 히스토리
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create training dataset
    train_dataset, train_dataloader = create_training_dataset(
        train_csv_path, train_img_path, processor, batch_size
    )
    
    # Train model
    epoch_losses = train_model(
        model, train_dataloader, device, num_epochs, learning_rate
    )
    
    # Save model and processor
    save_model_and_processor(model, processor, model_save_path, processor_save_path)
    
    return epoch_losses

def load_trained_model(model_path, processor_path, device):
    """
    학습된 모델과 프로세서를 로드하는 함수
    
    Args:
        model_path (str): 모델 저장 경로
        processor_path (str): 프로세서 저장 경로
        device (torch.device): 사용할 디바이스
    
    Returns:
        tuple: (model, processor)
    """
    print(f"Loading model from {model_path}")
    model = BeitForImageClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    print(f"Loading processor from {processor_path}")
    processor = BeitImageProcessor.from_pretrained(processor_path)
    
    return model, processor

def predict_test_data(model, processor, submission_csv_path, test_img_path, device, file_suffix="predictions"):
    """
    테스트 데이터에 대해 예측을 수행하는 함수
    
    Args:
        model: 로드된 모델
        processor: 로드된 프로세서
        test_csv_path (str): 테스트 CSV 파일 경로
        test_img_path (str): 테스트 이미지 폴더 경로
        submission_csv_path (str): 샘플 제출 파일 경로
        device (torch.device): 사용할 디바이스
        file_suffix (str): 결과 파일명에 추가할 접미사
    
    Returns:
        str: 저장된 예측 파일 경로
    """
    print("Creating test dataset and dataloader...")
    
    # Create test dataset and dataloader (no augmentation for test data)
    test_dataset = ImageDataset(submission_csv_path, test_img_path, processor, is_train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    print(f"Predicting on {len(test_dataset)} test samples...")
    
    # Lists to store predictions
    all_predictions = []
    
    # Predict without gradient computation
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Predicting"):
            pixel_values = batch["pixel_values"].to(device)
            
            # Get model predictions
            outputs = model(pixel_values=pixel_values)
            predictions = outputs.logits.argmax(-1).cpu().numpy()
            
            all_predictions.extend(predictions)
    
    print(f"Predictions completed. Total predictions: {len(all_predictions)}")

    # Read sample submission file
    submission_df = pd.read_csv(submission_csv_path)
    
    # Update target column with predictions
    submission_df['target'] = all_predictions
    
    # Save predictions to csv
    output_path = f"data/sample_submission_{file_suffix}.csv"
    submission_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    
    return output_path

def train_and_predict_pipeline(is_train=True):
    """
    훈련부터 예측까지 전체 파이프라인을 실행하는 함수
    """
    print("=== Starting Training or Prediction Pipeline ===")
    # 모델 저장 디렉토리 지정 이후 예측 시 모델 로드할때 사용
    model_save_path = "/data/ephemeral/home/Doc_Classification/models/finetuned_beit_A_model"
    processor_save_path = "/data/ephemeral/home/Doc_Classification/models/finetuned_beit_A_processor"

    if is_train:
        # 1. 모델 훈련
        print("\n--- Step 1: Model Training ---")
        epoch_losses = train_and_save_model(
            model=model,
            processor=processor,
            train_csv_path="data/train.csv",
            train_img_path="data/train",
            model_save_path=model_save_path,
            processor_save_path=processor_save_path,
            num_epochs=10,
            learning_rate=5e-5,
            batch_size=8
        )
        
        print(f"Training completed. Final loss: {epoch_losses[-1]:.4f}")
    else:
        # 2. 모델 로딩 및 예측
        print("\n--- Step 2: Model Prediction ---")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        loaded_model, loaded_processor = load_trained_model(
            model_save_path, processor_save_path, device
        )
        
        output_file = predict_test_data(
            model=loaded_model,
            processor=loaded_processor,
            submission_csv_path="data/sample_submission.csv",
            test_img_path="data/test",
            device=device,
            file_suffix="beit_A_pipeline"
        )
        
        print(f"\n=== Pipeline completed successfully! ===")
        print(f"Results saved to: {output_file}")
        

# 메인 함수 실행
if __name__ == "__main__":
    # 모델 훈련 모드 is_train=True
    train_and_predict_pipeline(is_train=True) 
    # 모델 예측 모드 is_train=False
    train_and_predict_pipeline(is_train=False)
