import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.dataset.dataset import DeepfakeDataset
from src.models.videomae import VideoMAEClassifier

def get_attention_maps(model, x):
    # Pass through the VideoMAE backbone
    outputs = model.model.videomae(pixel_values=x)
    
    # last_hidden_state shape: (batch_size, num_patches, hidden_size)
    # Using the L2 norm (magnitude) of the patch embeddings as a proxy for attention/activation
    hidden_states = outputs.last_hidden_state
    
    # Shape: (batch_size, num_patches)
    patch_activations = torch.norm(hidden_states, p=2, dim=-1)
    
    return patch_activations

def visualize_attention():
    os.makedirs("outputs/visualizations", exist_ok=True)
    
    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = VideoMAEClassifier().to(device)
    model.load_state_dict(torch.load("outputs/checkpoints/best_model.pth", map_location=device))
    model.eval()
    
    # Load dataset
    dataset = DeepfakeDataset("data/faces", multi_view=False)
    
    # Pick a random fake video and a random real video
    real_idx, fake_idx = -1, -1
    for i in range(len(dataset)):
        if dataset.samples[i][1] == 0 and real_idx == -1:
            real_idx = i
        if dataset.samples[i][1] == 1 and fake_idx == -1:
            fake_idx = i
        if real_idx != -1 and fake_idx != -1:
            break
            
    for idx, label_name in [(real_idx, "Real"), (fake_idx, "Fake")]:
        frames, label = dataset[idx] # frames: (T, C, H, W)
        x = frames.unsqueeze(0).to(device) # (1, T, C, H, W)
        
        with torch.no_grad():
            patch_attentions = get_attention_maps(model, x) # (1, 1568)
            
        patch_attentions = patch_attentions[0].cpu().numpy()
        
        # Reshape to (T_patches, H_patches, W_patches) = (8, 14, 14)
        # Assuming 16 frames, tubelet size 2x16x16, spatial size 224x224
        heatmaps = patch_attentions.reshape(8, 14, 14)
        
        # Normalize heatmaps to 0-1
        heatmaps = (heatmaps - heatmaps.min()) / (heatmaps.max() - heatmaps.min() + 1e-8)
        
        # Unnormalize original frames for plotting
        frames_np = frames.permute(0, 2, 3, 1).numpy() # (T, H, W, C)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        frames_np = np.clip((frames_np * std + mean) * 255.0, 0, 255).astype(np.uint8)
        
        fig, axes = plt.subplots(4, 4, figsize=(15, 15))
        fig.suptitle(f"Attention Maps for {label_name} Video", fontsize=20)
        
        for i in range(16):
            ax = axes[i // 4, i % 4]
            # Tubelet i covers frames 2*t and 2*t+1. So frame i corresponds to heatmap i//2
            heatmap = heatmaps[i // 2]
            
            # Resize heatmap to 224x224
            heatmap_resized = cv2.resize(heatmap, (224, 224))
            
            # Create color map
            heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            
            # Overlay
            overlay = cv2.addWeighted(frames_np[i], 0.6, heatmap_colored, 0.4, 0)
            
            ax.imshow(overlay)
            ax.axis("off")
            ax.set_title(f"Frame {i+1}")
            
        plt.tight_layout()
        plt.savefig(f"outputs/visualizations/attention_{label_name.lower()}.png")
        plt.close()
        
if __name__ == "__main__":
    visualize_attention()
    print("Attention maps saved in outputs/visualizations/")
