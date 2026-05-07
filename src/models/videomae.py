import torch
import torch.nn as nn
from transformers import VideoMAEForVideoClassification


class VideoMAEClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        # Pretrained True VideoMAE backbone
        self.model = VideoMAEForVideoClassification.from_pretrained(
            "MCG-NJU/videomae-base",
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        
        # Projection Head for Domain SSL (Contrastive Learning)
        self.projection_head = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

    def forward(self, x):
        # x shape: (B, T, C, H, W)
        
        # Extract features from the base VideoMAE encoder
        outputs = self.model.videomae(pixel_values=x)
        
        # Mean pool over the spatio-temporal patches
        # last_hidden_state shape: (B, num_patches, 768)
        video_features = outputs.last_hidden_state.mean(dim=1) 
        
        # 1. Classification Logits
        logits = self.model.classifier(video_features)
        
        # 2. SSL Embeddings
        embeddings = self.projection_head(video_features)
        
        return logits, embeddings


