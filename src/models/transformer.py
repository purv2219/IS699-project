#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Gesture Transformer Model
------------------------
Transformer-based model for sign language gesture recognition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class GestureTransformer(nn.Module):
    def __init__(self, num_classes, feature_dim=2048, seq_len=50, hidden_dim=768):
        super(GestureTransformer, self).__init__()

        self.input_proj = nn.Linear(feature_dim, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8),
            num_layers=6
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x.permute(1, 0, 2))  # (seq_len, batch, hidden_dim)
        x = x.mean(dim=0)  # Pooling over sequence
        return self.fc(x)
    
    def predict(self, features, device=None):
        """
        Make prediction for a sequence of features.
        
        Args:
            features (numpy.ndarray): Array of shape (seq_len, feature_dim)
            device (torch.device): Device to run the model on
            
        Returns:
            int: Index of the predicted class
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        # Prepare input tensor (pad or truncate to max_seq_len)
        max_seq_len = 50
        feature_dim = 2048
        
        padded_features = np.zeros((max_seq_len, feature_dim))
        length = min(len(features), max_seq_len)
        padded_features[:length] = features[:length]
        
        # Convert to tensor and pass through model
        input_tensor = torch.tensor(padded_features, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = self(input_tensor)
            predicted_idx = torch.argmax(logits, dim=1).item()
            
        return predicted_idx 