import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from timm import create_model
from torchvision import transforms
from tqdm import tqdm
from torch.nn.functional import softmax
import wandb
from datetime import datetime, timezone, timedelta
import glob
import json

KST = timezone(timedelta(hours=9))

def load_model(model_name, num_classes, ckpt_path, device, img_size=None):
    model = create_model(
        model_name,
        pretrained=False,
        num_classes=num_classes,
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_soft_voting(models, image_tensor, device):
    probs = []
    with torch.no_grad():
        for model in models:
            out = model(image_tensor)
            probs.append(softmax(out, dim=1).cpu().numpy())
    return np.mean(probs, axis=0)

CONFUSED_PAIRS = [(3, 7), (4,7), (4, 14), (3, 14), (6,10), (10,12), (13,14)]

def correct_confused(pred, prob_vec, confused_pairs=CONFUSED_PAIRS, threshold=0.1):
    for a, b in confused_pairs:
        if pred in (a, b):
            prob_a, prob_b = prob_vec[a], prob_vec[b]
            if abs(prob_a - prob_b) < threshold:
                return a if prob_a > prob_b else b
    return pred

def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb.init(project="cnn-doc-classification", name="2stage_inference_submission")

    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # ⬇️ doc target_map 로드 (역매핑)
    with open(f"{args.model_name}_doc_target_map.json", "r") as f:
        reverse_doc_map = {v: int(k) for k, v in json.load(f).items()}

    # Load binary classifier
    binary_model = load_model(args.model_name, 2, args.binary_model_path, device, args.img_size)

    # Load doc models
    doc_models = []
    for fold in args.select_folds:
        pattern = f"*{args.model_name}_doc_fold{fold}.pth"
        matched = glob.glob(pattern)
        if matched:
            doc_models.append(load_model(args.model_name, 15, matched[0], device, args.img_size))
        else:
            raise FileNotFoundError(f"No model found matching: {pattern}")

    # Load non-doc models
    non_doc_models = []
    for fold in args.select_folds:
        pattern = f"*{args.model_name}_non_doc_fold{fold}.pth"
        matched = glob.glob(pattern)
        if matched:
            non_doc_models.append(load_model(args.model_name, 2, matched[0], device, args.img_size))
        else:
            raise FileNotFoundError(f"No model found matching: {pattern}")

    test_dir = os.path.join(args.data_dir, "test")
    test_files = sorted(os.listdir(test_dir))

    preds = []
    doc_count = 0
    non_doc_count = 0

    for file in tqdm(test_files):
        img = Image.open(os.path.join(test_dir, file)).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = binary_model(img_tensor)
            prob = softmax(output, dim=1)[0]  # [비문서 확률, 문서 확률]
            is_doc = prob[1] > 0.6  # 문서일 확률이 60% 이상이면 문서로 판단

        if is_doc:
            doc_count += 1
            prob = predict_soft_voting(doc_models, img_tensor, device)[0]
            pred = np.argmax(prob)
            pred = correct_confused(pred, prob)
            pred = reverse_doc_map[pred]  # ⬅️ 역매핑
        else:
            non_doc_count += 1
            prob = predict_soft_voting(non_doc_models, img_tensor, device)[0]
            pred = 2 if prob[0] > prob[1] else 16

        preds.append((file, pred))

    sub = pd.read_csv(os.path.join(args.data_dir, "sample_submission.csv"))
    sub = sub.set_index("ID")
    for file, pred in preds:
        sub.loc[file, "target"] = int(pred)

    output_path = f"../output/{datetime.now(KST).strftime('%Y%m%d_%H%M%S')}_two_stage_submission.csv"
    sub.reset_index().to_csv(output_path, index=False)
    print(f"\n✅ 저장 완료: {output_path}")

    artifact = wandb.Artifact("2stage_submission", type="predictions")
    artifact.add_file(output_path)
    wandb.log_artifact(artifact)

    wandb.log({
        "doc_count": doc_count,
        "non_doc_count": non_doc_count,
        "total_test_samples": len(test_files)
    })

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../input/data')
    parser.add_argument('--model_name', type=str, default='convnext_base')
    parser.add_argument('--img_size', type=int, default=384)
    parser.add_argument('--binary_model_path', type=str, default='binary_non_doc_classifier.pth')
    parser.add_argument('--select_folds', type=int, nargs='+', default=[0, 1])
    args = parser.parse_args()

    run_inference(args)
