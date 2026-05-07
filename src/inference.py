import torch
import cv2
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
from PIL import Image

from src.models.videomae import VideoMAEClassifier

device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

# Load model globally to avoid reloading per request
model = VideoMAEClassifier().to(device)
model.load_state_dict(torch.load("outputs/checkpoints/best_model.pth", map_location=device))
model.eval()

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

# Initialize OpenCV Face Tracker
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def extract_faces_from_video(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip = max(1, total_frames // num_frames) if total_frames > 0 else 1
    
    count = 0
    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        if count % skip == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                # Get the largest face
                x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
                # Expand bounding box slightly for context
                padding = int(w * 0.2)
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(frame.shape[1], x + w + padding)
                y2 = min(frame.shape[0], y + h + padding)
                
                face_crop = frame[y1:y2, x1:x2]
            else:
                # If no face found, use the center crop as fallback
                h, w = frame.shape[:2]
                min_dim = min(h, w)
                face_crop = frame[(h-min_dim)//2:(h+min_dim)//2, (w-min_dim)//2:(w+min_dim)//2]
                
            face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            frames.append(face_crop)
            
        count += 1
        
    cap.release()
    
    while len(frames) < num_frames and len(frames) > 0:
        frames.append(frames[-1])
        
    if len(frames) == 0:
        frames = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(num_frames)]
        
    return frames

def preprocess_frames(frames_np):
    processed = []
    for img in frames_np:
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        img = normalize(img)
        processed.append(img)
    return torch.stack(processed).unsqueeze(0).to(device)

def get_activations(model, x):
    with torch.no_grad():
        outputs = model.model.videomae(pixel_values=x)
        hidden_states = outputs.last_hidden_state
        patch_activations = torch.norm(hidden_states, p=2, dim=-1) # (1, 1568)
    return patch_activations[0].cpu().numpy()

def generate_heatmap_and_graph(patch_attentions, frames_np):
    heatmaps = patch_attentions.reshape(8, 14, 14)
    heatmaps = (heatmaps - heatmaps.min()) / (heatmaps.max() - heatmaps.min() + 1e-8)
    
    # 1. Overlay Heatmap on middle frame
    idx = 8 
    heatmap = heatmaps[idx // 2]
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    original_frame = cv2.resize(frames_np[idx], (224, 224))
    overlay = cv2.addWeighted(original_frame, 0.6, heatmap_colored, 0.4, 0)
    
    # 2. Generate Suspicion Graph
    temporal_scores = heatmaps.mean(axis=(1, 2)) # Shape: (8,)
    # Duplicate for 16 frames
    frame_scores = np.repeat(temporal_scores, 2)
    # Normalize scores to 0-100%
    frame_scores = (frame_scores / frame_scores.max()) * 100.0
    
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(1, 17), frame_scores, color='salmon')
    ax.set_title("Per-Frame Suspicion Analysis (Attention Focus)")
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Suspicion / Attention Level (%)")
    ax.set_xticks(range(1, 17))
    ax.set_ylim(0, 110)
    
    # Color top suspicious frames deep red
    for i, bar in enumerate(bars):
        if frame_scores[i] > 85:
            bar.set_color('darkred')
            
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    graph_img = np.array(Image.open(buf))
    plt.close()
    
    return overlay, graph_img

def predict_deepfake(video_path):
    if not video_path:
        return {"Error": 1.0}, None, None
        
    frames_np = extract_faces_from_video(video_path)
    x = preprocess_frames(frames_np)
    
    with torch.no_grad():
        logits, _ = model(x)
        probs = F.softmax(logits, dim=1)[0]
        
    real_prob = probs[0].item()
    fake_prob = probs[1].item()
    
    confidences = {"Real": real_prob, "Fake": fake_prob}
    
    patch_attentions = get_activations(model, x)
    overlay_img, graph_img = generate_heatmap_and_graph(patch_attentions, frames_np)
    
    return confidences, overlay_img, graph_img
