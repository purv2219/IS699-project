import csv
import re
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import argparse
from torchvision import models, transforms
from PIL import Image

# ---------------------------------------------------------
# 1. Setup and Arguments
# ---------------------------------------------------------
labels_path = "data/labels/word_to_label.pkl"
if not os.path.exists(labels_path):
    print(f"Warning: Label file not found at {labels_path}. Prediction will show ID instead of Name.")
    word_to_label = {}
else:
    with open(labels_path, "rb") as f:
        word_to_label = pickle.load(f)

label_to_word = {v: k for k, v in word_to_label.items()}

parser = argparse.ArgumentParser(description="Sign Language Prediction")
parser.add_argument("-i", "--input", required=True,
                    help="Path to input folder (images) OR .pkl features file")
parser.add_argument("--key", default=None,
                    help="Which item to predict (dict key / video name / list index)")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


GT_CSV_PATH = "data/labels/test_words.csv"  

def load_ground_truth(csv_path):
    gt = {}
    if not os.path.exists(csv_path):
        print(f"Warning: Ground truth file not found at {csv_path}")
        return gt

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                vid = int(row["video"])
                word = row["word"].strip()
                start = int(row["startframe"])
            except Exception:
                continue
            gt.setdefault(vid, []).append((start, word))

    # sort by startframe and keep only words
    gt_sorted = {}
    for vid, items in gt.items():
        items.sort(key=lambda x: x[0])
        gt_sorted[vid] = [w for _, w in items]
    return gt_sorted

ground_truth = load_ground_truth(GT_CSV_PATH)
print("GT file used:", GT_CSV_PATH)
print("GT videos loaded:", len(ground_truth))


def infer_video_id(input_path, select_key):
    # best: key is numeric video id
    if select_key is not None and str(select_key).isdigit():
        return int(select_key)

    # else try digits from folder/file name (e.g., 002, video_2)
    base = os.path.basename(os.path.normpath(input_path))
    m = re.search(r"(\d+)", base)
    if m:
        return int(m.group(1))
    return None


# ---------------------------------------------------------
# 2. Model Definition
# ---------------------------------------------------------
class GestureTransformer(nn.Module):
    def __init__(self, num_classes, feature_dim=2048, hidden_dim=768):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)

num_classes = len(word_to_label) if word_to_label else 10
model = GestureTransformer(num_classes=num_classes).to(device)

model_path = "models/gesture_transformer.pth"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}")

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ---------------------------------------------------------
# 3. Helper Functions
# ---------------------------------------------------------
def predict_from_features(features_seq):
    max_seq_len = 50
    feature_dim = 2048

    if features_seq.ndim == 1:
        features_seq = features_seq.reshape(1, -1)

    padded = np.zeros((max_seq_len, feature_dim))
    length = min(len(features_seq), max_seq_len)
    padded[:length] = features_seq[:length]

    tensor = torch.tensor(padded, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        idx = torch.argmax(logits, dim=1).item()

    return label_to_word.get(idx, f"Unknown_ID_{idx}")

def extract_features_from_folder(folder_path):
    from torchvision.models import ResNet50_Weights
    print(f"Extracting features from images in: {folder_path}")

    resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    resnet.eval().to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    feats = []
    images = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ])

    if not images:
        raise ValueError(f"No images found in {folder_path}")

    for img in images:
        path = os.path.join(folder_path, img)
        image = Image.open(path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            f = resnet(image)
        feats.append(f.view(-1, 2048).cpu().numpy())

    return np.vstack(feats)

# ---------------------------------------------------------
# 4. Input Loader (MANUAL SELECTION FIX)
# ---------------------------------------------------------
input_path = args.input
select_key = args.key
features_seq = None

if os.path.isfile(input_path) and input_path.endswith(".pkl"):
    with open(input_path, "rb") as f:
        data = pickle.load(f)

    print(f"Loaded pickle type: {type(data)}")

    # -------- Dictionary --------
    if isinstance(data, dict):
        if select_key is None:
            raise ValueError(
                "Pickle contains multiple entries. "
                "Use --key <dict_key>"
            )
        if select_key not in data:
            raise KeyError(
                f"Key '{select_key}' not found. "
                f"Available keys (sample): {list(data.keys())[:5]}"
            )

        print(f"Using key: {select_key}")
        features_seq = data[select_key]

    # -------- List --------
    elif isinstance(data, list):
        if select_key is None:
            raise ValueError(
                "Pickle contains a list. "
                "Use --key <index or video_name>"
            )

        # index selection
        if select_key.isdigit():
            idx = int(select_key)
            if idx >= len(data):
                raise IndexError("Index out of range")
            item = data[idx]

        # name-based selection
        else:
            item = None
            for entry in data:
                if isinstance(entry, (list, tuple)) and select_key in entry:
                    item = entry
                    break
            if item is None:
                raise ValueError(f"No entry found for '{select_key}'")

        if isinstance(item, (list, tuple)):
            for sub in item:
                if isinstance(sub, np.ndarray):
                    features_seq = sub
                    break
        elif isinstance(item, np.ndarray):
            features_seq = item

    # -------- Direct Array --------
    elif isinstance(data, np.ndarray):
        features_seq = data

    if features_seq is None:
        raise ValueError("Failed to extract feature array from pickle")

elif os.path.isdir(input_path):
    features_seq = extract_features_from_folder(input_path)

else:
    raise ValueError("Input must be a directory or .pkl file")

# ---------------------------------------------------------
# 5. Prediction
# ---------------------------------------------------------
print(f"Feature shape: {features_seq.shape}")
prediction = predict_from_features(features_seq)
print(f"Predicted Gesture: {prediction}")

video_id = infer_video_id(input_path, select_key)

if video_id is None:
    print("Ground Truth: (Could not infer video id. Use --key <video_number> like --key 2)")
else:
    gt_words = ground_truth.get(video_id)
    if not gt_words:
        print(f"Ground Truth: (No GT found for video={video_id} in {GT_CSV_PATH})")
    else:
        print(f"Video ID: {video_id}")
        print("Ground Truth Words:", " ".join(gt_words))
