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
    
    def predict(self, features, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        max_seq_len = 50
        feature_dim = 2048
        
        padded_features = np.zeros((max_seq_len, feature_dim))
        length = min(len(features), max_seq_len)
        padded_features[:length] = features[:length]
        
        input_tensor = torch.tensor(padded_features, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = self(input_tensor)
            predicted_idx = torch.argmax(logits, dim=1).item()
        
        return predicted_idx
