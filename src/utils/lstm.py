import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertConfig
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
train=True
# Load TRAIN features
with open("features/resnet_features_train.pkl", "rb") as f:
    train_feature_data = pickle.load(f)

print("Loaded training feature data:", len(train_feature_data), "samples")

# Load TEST features (for validation)
with open("features/resnet_features_test.pkl", "rb") as f:
    test_feature_data = pickle.load(f)

print("Loaded test feature data:", len(test_feature_data), "samples")

# Create dictionary mapping words to labels (use TRAIN labels)
unique_words = sorted(set(entry[1] for entry in train_feature_data))  # Use training labels only
word_to_label = {word: idx for idx, word in enumerate(unique_words)}
with open("word_to_label.pkl", "wb") as f:
    pickle.dump(word_to_label, f)
# Define Dataset Classimport torch
import numpy as np
import random

class GestureDataset(Dataset):
    def __init__(self, feature_data, word_to_label, max_seq_len=50, feature_dim=2048, augment=False, class_counts=None):
        self.data = feature_data
        self.word_to_label = word_to_label
        self.max_seq_len = max_seq_len
        self.feature_dim = feature_dim
        self.augment = augment
        self.class_counts = class_counts  # Track occurrences of each class

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        _, word, features = self.data[idx]
        label = self.word_to_label.get(word, -1)

        # Pad or truncate features
        padded_features = np.zeros((self.max_seq_len, self.feature_dim))
        length = min(len(features), self.max_seq_len)
        padded_features[:length] = features[:length]  # Truncation
        actual_seq_len = min(len(features), self.max_seq_len)


        # Apply augmentation **only for underrepresented classes**
        # if self.augment and self.class_counts.get(word, 0) < 5:  # If fewer than 5 samples, augment
        padded_features = self.augment_features(padded_features)

        return torch.tensor(padded_features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def augment_features(self, features):
        """ Apply augmentations to underrepresented classes """
        if random.random() < 0.3:
            features = self.gaussian_noise(features)
        if random.random() < 0.3:
            features = self.time_masking(features)
        if random.random() < 0.3:
            features = self.frame_dropout(features)
        return features

    def gaussian_noise(self, features, mean=0, std=0.01):
        noise = np.random.normal(mean, std, features.shape)
        return features + noise

    def time_masking(self, features, mask_ratio=0.2):
        num_masks = int(self.max_seq_len * mask_ratio)
        mask_indices = np.random.choice(self.max_seq_len, num_masks, replace=False)
        features[mask_indices] = 0
        return features

    def frame_dropout(self, features, drop_prob=0.1):
        for i in range(1, self.max_seq_len):
            if random.random() < drop_prob:
                features[i] = features[i - 1]
        return features

from collections import Counter

word_counts = Counter(entry[1] for entry in train_feature_data)  # Count occurrences of each label
from imblearn.over_sampling import RandomOverSampler

# Convert dataset to NumPy array for re-sampling
X_train = [entry[2] for entry in train_feature_data]  # Features
y_train = [word_to_label[entry[1]] for entry in train_feature_data]  # Labels

# Apply over-sampling
ros = RandomOverSampler(sampling_strategy='auto')
X_resampled, y_resampled = ros.fit_resample(np.array(X_train, dtype=object).reshape(-1, 1), y_train)

# Reformat into dataset structure
label_to_word = {idx: word for word, idx in word_to_label.items()}  # Invert mapping
resampled_data = [(None, label_to_word[label], feature) for feature, label in zip(X_resampled.flatten(), y_resampled)]

train_dataset = GestureDataset(resampled_data, word_to_label, augment=True, class_counts=word_counts)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Load dataset with class-aware augmentation
# train_dataset = GestureDataset(train_feature_data, word_to_label, augment=True, class_counts=word_counts)
test_dataset = GestureDataset(test_feature_data, word_to_label, augment=False)

# train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        attn_weights = torch.softmax(self.attn(x), dim=1)  # (batch, seq_len, 1)
        return torch.sum(attn_weights * x, dim=1)  # Weighted sum over sequence

class GestureBiLSTMAttention(nn.Module):
    def __init__(self, num_classes, feature_dim=2048, seq_len=50, hidden_dim=768):
        super(GestureBiLSTMAttention, self).__init__()
        
        self.input_proj = nn.Linear(feature_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim // 2, num_layers=2, bidirectional=True, batch_first=True)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = self.input_proj(x)
        x, _ = self.lstm(x)  # (batch, seq_len, hidden_dim)
        x = self.attention(x)  # (batch, hidden_dim)
        return self.fc(x)

# Initialize Model
num_classes = len(unique_words)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GestureBiLSTMAttention(num_classes=num_classes).to(device)

# Define Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-6)
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Track time
start_time = time.time()

# Store loss values for plotting
train_losses = []
val_losses = []
val_accuracies = []
num_epochs = 50
if train:
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for features, labels in train_dataloader:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        train_loss = total_loss / len(train_dataloader)
        train_losses.append(train_loss)  # Store training loss
        
        # Validation Step
        model.eval()
        all_labels, all_preds = [], []
        val_loss = 0

        with torch.no_grad():
            for features, labels in test_dataloader:
                features, labels = features.to(device), labels.to(device)

                mask = labels != -1
                if not mask.any():
                    continue

                features, labels = features[mask], labels[mask]
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        val_loss /= len(test_dataloader)
        val_accuracy = accuracy_score(all_labels, all_preds)

        val_losses.append(val_loss)  # Store validation loss
        val_accuracies.append(val_accuracy)  # Store validation accuracy

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # Save Model
    torch.save(model.state_dict(), "models/gesture_transformer.pth")

# Compute total training time
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Total Training Time: {training_time:.2f} seconds")

    # Save training statistics to a file
    with open("results/training_stats.txt", "w") as f:
        f.write(f"Total Training Time: {training_time:.2f} seconds/n")
        for epoch in range(num_epochs):
            f.write(f"Epoch {epoch+1}: Train Loss = {train_losses[epoch]:.4f}, Val Loss = {val_losses[epoch]:.4f}, Val Accuracy = {val_accuracies[epoch]:.4f}/n")

    # Plot Training and Validation Loss Curve
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss Curve")
    plt.legend()
    plt.grid()
    plt.savefig("results/loss_curve.png")
    plt.show()

# Final Model Evaluation
model.load_state_dict(torch.load("models/gesture_transformer.pth"))
model.eval()

all_labels, all_preds = [], []
total_loss = 0

with torch.no_grad():
    for features, labels in test_dataloader:
        features, labels = features.to(device), labels.to(device)

        mask = labels != -1
        if not mask.any():
            continue

        features, labels = features[mask], labels[mask]
        outputs = model(features)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

test_loss = total_loss / len(test_dataloader)
test_accuracy = accuracy_score(all_labels, all_preds)

print(f"Final Test Accuracy: {test_accuracy:.4f}")
print(f"Final Test Loss: {test_loss:.4f}")

# Classification Report and Confusion Matrix
unique_test_labels = sorted(set(all_labels))
label_names = [unique_words[i] for i in unique_test_labels]

classification_rep = classification_report(all_labels, all_preds, labels=unique_test_labels, target_names=label_names)
conf_matrix = confusion_matrix(all_labels, all_preds, labels=unique_test_labels)

print("/nClassification Report:")
print(classification_rep)

print("Confusion Matrix:/n", conf_matrix)

# Save evaluation results
with open("results/evaluation.txt", "w") as f:
    f.write(f"Final Test Accuracy: {test_accuracy:.4f}/n")
    f.write(f"Final Test Loss: {test_loss:.4f}/n/n")
    f.write("Classification Report:/n")
    f.write(classification_rep + "/n/n")
    f.write("Confusion Matrix:/n")
    f.write(np.array2string(conf_matrix) + "/n")

# Plot Confusion Matrix
plt.figure(figsize=(12, 8))
df_cm = pd.DataFrame(conf_matrix, index=label_names, columns=label_names)
sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues", xticklabels=True, yticklabels=True)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.savefig("results/confusion_matrix.png")
plt.show()
