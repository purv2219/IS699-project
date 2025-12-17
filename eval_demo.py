import csv
import os
import pickle
import argparse
import re

import torch
import torch.nn as nn
import numpy as np


# ------------ model (same as your predict.py) ------------
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


def load_ground_truth(csv_path: str) -> dict:
    """
    Returns:
      { video_id(int): [word1, word2, ...] }  sorted by startframe
    """
    gt = {}
    if not os.path.exists(csv_path):
        print(f"[ERROR] GT not found: {csv_path}")
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

    out = {}
    for vid, items in gt.items():
        items.sort(key=lambda x: x[0])
        out[vid] = [w for _, w in items]
    return out


def normalize_features(item):
    """
    Your pickle sometimes stores list items as:
      - np.ndarray
      - (something, np.ndarray)
      - [something, np.ndarray, ...]
    This function always returns a numpy array (frames, 2048).
    """
    if isinstance(item, np.ndarray):
        return item

    if isinstance(item, (tuple, list)):
        for part in item:
            if isinstance(part, np.ndarray):
                return part

    raise ValueError(f"Could not find numpy array in item. Type={type(item)}")


def predict_top1(model, device, features_seq: np.ndarray) -> int:
    max_seq_len = 50
    feature_dim = 2048

    if features_seq.ndim == 1:
        features_seq = features_seq.reshape(1, -1)

    padded = np.zeros((max_seq_len, feature_dim), dtype=np.float32)
    length = min(len(features_seq), max_seq_len)
    padded[:length] = features_seq[:length]

    x = torch.tensor(padded, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        idx = torch.argmax(logits, dim=1).item()

    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="features pkl (list or dict)")
    ap.add_argument("--gt", default="data/labels/test_words.csv")
    ap.add_argument("--model", default="models/gesture_transformer.pth")
    ap.add_argument("--labels", default="data/labels/word_to_label.pkl")
    ap.add_argument("--out", default="demo_results.csv")
    ap.add_argument("--limit", type=int, default=60, help="how many items to evaluate")
    args = ap.parse_args()

    # labels
    if not os.path.exists(args.labels):
        raise FileNotFoundError(f"Label map not found: {args.labels}")

    with open(args.labels, "rb") as f:
        word_to_label = pickle.load(f)

    label_to_word = {v: k for k, v in word_to_label.items()}
    num_classes = len(word_to_label)

    # model
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GestureTransformer(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    # data
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Features pkl not found: {args.input}")

    with open(args.input, "rb") as f:
        data = pickle.load(f)

    # ground truth
    gt = load_ground_truth(args.gt)
    print("GT file used:", args.gt)
    print("GT videos loaded:", len(gt))

    rows = []

    # -------- list case --------
    if isinstance(data, list):
        n = min(len(data), args.limit)
        for vid in range(1, n + 1):
            item = data[vid - 1]
            feat = normalize_features(item)

            pred_idx = predict_top1(model, device, feat)
            pred_word = label_to_word.get(pred_idx, f"Unknown_ID_{pred_idx}")

            gt_words = gt.get(vid, [])
            rows.append([vid, pred_word, " ".join(gt_words)])

    # -------- dict case --------
    elif isinstance(data, dict):
        keys = list(data.keys())[: args.limit]
        for k in keys:
            item = data[k]
            feat = normalize_features(item)

            m = re.search(r"(\d+)", str(k))
            vid = int(m.group(1)) if m else -1

            pred_idx = predict_top1(model, device, feat)
            pred_word = label_to_word.get(pred_idx, f"Unknown_ID_{pred_idx}")

            gt_words = gt.get(vid, [])
            rows.append([vid, pred_word, " ".join(gt_words)])

    else:
        raise TypeError(f"Unsupported pickle type: {type(data)}")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "predicted_top1", "ground_truth_words"])
        w.writerows(rows)

    print(f"[OK] Saved: {args.out}")


if __name__ == "__main__":
    main()
