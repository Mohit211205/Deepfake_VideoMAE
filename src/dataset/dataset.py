import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from torchvision import transforms
import random


class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, num_frames=16, multi_view=True):
        self.samples = []
        self.num_frames = num_frames
        self.multi_view = multi_view
        
        # VideoMAE/ImageNet normalization stats
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        self.color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
        self.blur = transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))


        for label, cls in enumerate(["real", "fake"]):
            class_path = os.path.join(root_dir, cls)

            for video in os.listdir(class_path):
                video_path = os.path.join(class_path, video)
                self.samples.append((video_path, label))

    def __len__(self):
        return len(self.samples)

    def load_frames(self, video_path):
        frame_files = sorted(os.listdir(video_path))

        # Ensure fixed number of frames
        if len(frame_files) >= self.num_frames:
            indices = list(range(self.num_frames))
        else:
            indices = list(range(len(frame_files)))

        frames = []
        for i in indices:
            frame_path = os.path.join(video_path, frame_files[i])
            img = cv2.imread(frame_path)

            if img is None:
                continue

            # BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            img = cv2.resize(img, (224, 224))
            img = img / 255.0

            img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
            img = self.normalize(img)
            
            frames.append(img)

        # Pad if needed — guard against empty folders
        if len(frames) == 0:
            return frames  # caller will skip this sample
        while len(frames) < self.num_frames:
            frames.append(frames[-1])

        frames_tensor = torch.stack(frames) # Shape: (T, C, H, W)
        
        if not self.multi_view:
            return frames_tensor
            
        # ── WEAK VIEW: minimal perturbation, preserves semantics ──────────
        weak_frames = frames_tensor.clone()
        if random.random() > 0.5:
            weak_frames = weak_frames.flip(-1)  # Horizontal flip

        # ── STRONG VIEW: aggressive perturbation ───────────────────────────
        strong_frames = frames_tensor.clone()

        # 1. Spatial: Horizontal flip
        if random.random() > 0.5:
            strong_frames = strong_frames.flip(-1)

        # 2. Spatial: Vertical flip (rare)
        if random.random() > 0.8:
            strong_frames = strong_frames.flip(-2)

        # 3. Temporal: Shuffle a random 4-frame window
        if random.random() > 0.5:
            t = strong_frames.shape[0]
            start = random.randint(0, t - 4)
            idx = list(range(start, start + 4))
            random.shuffle(idx)
            perm = list(range(t))
            for i, j in zip(range(start, start + 4), idx):
                perm[i] = j
            strong_frames = strong_frames[perm]

        # 4. Color & Blur
        strong_frames = self.color_jitter(strong_frames)
        if random.random() > 0.4:
            strong_frames = self.blur(strong_frames)

        # 5. JPEG Compression artifacts — simulates social media re-encoding
        #    This is a KEY augmentation for deepfake detection generalization!
        if random.random() > 0.4:
            quality = random.randint(40, 85)  # Low quality = heavy compression
            compressed = []
            for frame in strong_frames:
                # Decode back to numpy, encode as JPEG, decode again
                np_frame = (frame.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
                _, encoded = cv2.imencode('.jpg', np_frame, encode_param)
                decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
                compressed.append(torch.tensor(decoded).permute(2, 0, 1))
            strong_frames = torch.stack(compressed)

        # 6. Gaussian Noise — simulates camera sensor noise / low light
        if random.random() > 0.5:
            noise = torch.randn_like(strong_frames) * random.uniform(0.01, 0.04)
            strong_frames = (strong_frames + noise).clamp(0, 1)

        # 7. Random Grayscale — forces model to rely on texture not color
        if random.random() > 0.85:
            gray = strong_frames.mean(dim=1, keepdim=True).expand_as(strong_frames)
            strong_frames = gray

        return weak_frames, strong_frames

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        frames = self.load_frames(video_path)

        # Skip empty folders gracefully
        if len(frames) == 0:
            # Return a dummy black frame batch so DataLoader doesn't crash
            dummy = torch.zeros(self.num_frames, 3, 224, 224)
            if self.multi_view:
                return dummy, dummy, torch.tensor(label)
            return dummy, torch.tensor(label)

        if self.multi_view:
            weak_frames, strong_frames = frames
            return weak_frames, strong_frames, torch.tensor(label)
        else:
            return frames, torch.tensor(label)
