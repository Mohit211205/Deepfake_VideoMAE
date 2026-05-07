import torch
from src.models.videomae import VideoMAEClassifier

model = VideoMAEClassifier()

x = torch.randn(2, 16, 3, 224, 224)  # batch=2

out = model(x)

print("Output shape:", out.shape)
