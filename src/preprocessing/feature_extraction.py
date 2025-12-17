import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os
import pandas as pd
import numpy as np
import pickle
import argparse

# Argument parser
parser = argparse.ArgumentParser(description="Extract features from images using ResNet50")
parser.add_argument("-i", "--input", required=True, help="Input directory containing video folders")
parser.add_argument("-t", "--type", choices=["train", "test"], required=True, help="Dataset type: train or test")
args = parser.parse_args()

# Load pre-trained ResNet model
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Select ResNet variant (ResNet18, ResNet34, ResNet50, ResNet101, ResNet152)
def get_resnet(model_name="resnet50"):
    resnet_variants = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "resnet152": models.resnet152
    }
    assert model_name in resnet_variants, f"Invalid ResNet model: {model_name}"
    
    model = resnet_variants[model_name](pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove FC layer
    return model

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Choose ResNet variant
resnet_model = get_resnet("resnet50")  # Change to resnet18, resnet34, etc.
resnet_model.to(device)
resnet_model.eval()

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_features(image_path):
    """Extract feature vector from an image using the chosen ResNet variant"""
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        features = resnet_model(img).squeeze().cpu().numpy()  # Extract feature vector
    return features

# Example usage
# feature_vector = extract_features("path/to/image.jpg")


# Load dataset CSV
import os  # make sure this is at the top of the file once

csv_path = os.path.join("data", "labels", f"{args.type}_words.csv")

df = pd.read_csv(csv_path)  # Load train or test CSV based on input

all_features = []  # Store all extracted features

# Iterate over all video folders (001 to 201)
for i in range(1, 202):  # Fix: Ensure range covers 001 to 201
    folder_name = f"{i:03d}"  # Convert to three-digit format
    frame_dir = os.path.join(args.input, folder_name, "mask")  # Directory containing masked frames

    if not os.path.exists(frame_dir):
        print(f"❌ Warning: Mask folder {frame_dir} does not exist. Skipping...")
        continue

    for _, row in df.iterrows():
        video_id, start, end, word = row["video"], row["startframe"], row["endframe"], row["word"]
        video_features = []

        if video_id == i:  # Process only matching video ID
            for frame_num in range(start, end + 1):
                frame_name = f"frame{frame_num:06d}_cam0_mask.png"  # Matches "frame000000_cam0_mask.png"
                frame_path = os.path.join(frame_dir, frame_name)

                if os.path.exists(frame_path):
                    features = extract_features(frame_path)
                    video_features.append(features)

            if video_features:  # Fix: Check before storing
                all_features.append((video_id, word, np.array(video_features, dtype=np.float32)))

    print(f"✅ Done processing folder {i:03d}")

# Save all extracted features into a file
output_file = f"C:/Users/purvs/Downloads/sign-language-recognition/features/resnet_features_{args.type}.pkl"
os.makedirs(os.path.dirname(output_file), exist_ok=True)  # Fix: Ensure output dir exists

with open(output_file, "wb") as f:
    pickle.dump(all_features, f)

print(f"🎉 All features saved successfully as {output_file}!")
