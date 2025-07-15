# vit_infer_fold.py
import argparse
import os
from datetime import datetime, timezone, timedelta

import pandas as pd
import torch
import timm
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Test dataset class
class TestDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(os.path.join(self.img_dir, row['ID'])).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, row['ID']

# Inference
@torch.no_grad()
def inference(model, loader, device):
    model.eval()
    preds, ids = [], []
    for images, id_list in tqdm(loader):
        images = images.to(device)
        outputs = model(images)
        pred = outputs.softmax(dim=1).argmax(dim=1).cpu().numpy()
        preds.extend(pred)
        ids.extend(id_list)
    return ids, preds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, required=True, help='Fold number (0-4)')
    parser.add_argument('--weight_path', type=str, required=True)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--data_dir', type=str, default='../input/data')
    parser.add_argument('--exp_name', type=str, default='vit_infer')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_csv = os.path.join(args.data_dir, 'sample_submission.csv')
    test_img_dir = os.path.join(args.data_dir, 'test')
    test_dataset = TestDataset(test_csv, test_img_dir, transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=17)
    model.load_state_dict(torch.load(args.weight_path, map_location=device))
    model.to(device)

    ids, preds = inference(model, test_loader, device)
    submission = pd.DataFrame({'ID': ids, 'target': preds})

    KST = timezone(timedelta(hours=9))
    timestamp = datetime.now(KST).strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), "output"))
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{timestamp}_{args.exp_name}_fold{args.fold}.csv")
    submission.to_csv(filename, index=False)
    print(f"✅ Inference 완료. 저장 위치: {filename}")

if __name__ == '__main__':
    main()
