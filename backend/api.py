import os

import io
import uuid
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor

_executor = ThreadPoolExecutor(max_workers=2)
import torch
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import base64
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
from facenet_pytorch import MTCNN
from src.models.videomae import VideoMAEClassifier

FEEDBACK_LOG     = "feedback_log.json"
FAKE_THRESHOLD   = 0.55   # Must cross this to be called FAKE (reduced false positives)
SUSPECT_THRESHOLD = 0.38  # Between SUSPECT and FAKE = "SUSPICIOUS" zone

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Backend using device: {device}")

# Download model weights from HF Hub if not present locally
MODEL_PATH = "outputs/checkpoints/best_model.pth"
if not os.path.exists(MODEL_PATH):
    print("Downloading model weights from HF Hub...")
    from huggingface_hub import hf_hub_download
    os.makedirs("outputs/checkpoints", exist_ok=True)
    hf_hub_download(
        repo_id="AtrriJi/deepfake-videomae-weights",
        filename="best_model.pth",
        local_dir="outputs/checkpoints",
    )
    print("Model weights downloaded!")

# Load VideoMAE Model
model = VideoMAEClassifier().to(device)
state_dict = torch.load(MODEL_PATH, map_location=device)

# Robust key mapper: handles both old (Phase 3 — no projection_head) and
# new (Phase 4+ — with projection_head) checkpoints.
new_state_dict = {}
for k, v in state_dict.items():
    new_k = k.replace('.attention.query.bias', '.attention.q_bias')
    new_k = new_k.replace('.attention.value.bias', '.attention.v_bias')
    if '.attention.key.bias' in new_k:
        continue
    new_state_dict[new_k] = v

# strict=False: safely ignores missing projection_head keys in old checkpoints
result = model.load_state_dict(new_state_dict, strict=False)
print(f"Checkpoint loaded. Missing keys: {len(result.missing_keys)} | Unexpected: {len(result.unexpected_keys)}")
model.eval()

# Load MTCNN Face Tracker
mtcnn = MTCNN(keep_all=False, device=device)

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)


def forensics_boost(frames, base_fake_prob):
    """ELA + DCT forensic analysis to boost fake detection for thin-layer deepfakes."""
    ela_scores = []
    dct_scores = []

    for frame in frames[:8]:  # Analyse first 8 frames for speed
        try:
            img_pil = Image.fromarray(frame)
            # ELA: re-compress and compare
            buf = io.BytesIO()
            img_pil.save(buf, format='JPEG', quality=90)
            buf.seek(0)
            recompressed = np.array(Image.open(buf).convert('RGB'))
            ela_diff = np.abs(frame.astype(np.float32) - recompressed.astype(np.float32))
            ela_scores.append(ela_diff.mean())

            # DCT: high-frequency energy in Y channel
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
            dct = cv2.dct(gray)
            h, w = gray.shape
            hf_energy = np.abs(dct[h//2:, w//2:]).mean()
            dct_scores.append(hf_energy)
        except Exception:
            pass

    ela_score = np.mean(ela_scores) if ela_scores else 0.0
    dct_score = np.mean(dct_scores) if dct_scores else 0.0

    # Thresholds (empirically tuned)
    ela_flag = ela_score > 8.0
    dct_flag = dct_score > 15.0

    boost = 0.0
    if ela_flag:
        boost += 0.07
    if dct_flag:
        boost += 0.05

    boosted = min(1.0, base_fake_prob + boost)
    return boosted, {"ela_score": round(ela_score, 3), "dct_score": round(dct_score, 3),
                     "ela_flag": ela_flag, "dct_flag": dct_flag}


def process_video_and_predict(video_bytes):
    # Use unique temp file to avoid race conditions (concurrent requests)
    temp_path = f"temp_{uuid.uuid4().hex}.mp4"

    try:
        with open(temp_path, "wb") as f:
            f.write(video_bytes)

        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            return {"error": "Cannot open video file. Try MP4 or WebM format."}

        frames = []
        num_frames = 16
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            return {"error": "Video has 0 frames. File may be corrupted."}

        # ── MTCNN face crop → 224x224 (matches training data exactly) ──────
        # Training data = 224x224 PRE-CROPPED face images.
        # Uploaded videos are full-resolution — face is tiny after resize.
        # Must crop face region first, THEN resize to 224x224.
        step = max(1, total_frames // num_frames)
        target_indices = set(min(i * step, total_frames - 1) for i in range(num_frames))
        count = 0
        from PIL import Image as PILImage

        all_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if count in target_indices:
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img   = PILImage.fromarray(frame_rgb)
                    boxes, _  = mtcnn.detect(pil_img)
                    if boxes is not None and len(boxes) > 0:
                        x1, y1, x2, y2 = [int(b) for b in boxes[0]]
                        pw = int((x2 - x1) * 0.2)
                        ph = int((y2 - y1) * 0.2)
                        x1 = max(0, x1 - pw);  y1 = max(0, y1 - ph)
                        x2 = min(frame_rgb.shape[1], x2 + pw)
                        y2 = min(frame_rgb.shape[0], y2 + ph)
                        crop = frame_rgb[y1:y2, x1:x2]
                        if crop.size == 0:
                            raise ValueError("empty crop")
                    else:
                        # No face detected: center-square crop
                        h, w = frame_rgb.shape[:2]
                        m = min(h, w)
                        crop = frame_rgb[(h-m)//2:(h+m)//2, (w-m)//2:(w+m)//2]
                    all_frames.append(cv2.resize(crop, (224, 224)))
                except Exception:
                    pass
            count += 1
        frames = all_frames[:num_frames]

        cap.release()
    except Exception as e:
        return {"error": f"Video processing failed: {str(e)}"}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    # Pad if not enough frames (same as training)
    while len(frames) < num_frames and len(frames) > 0:
        frames.append(frames[-1])

    if len(frames) == 0:
        return {"error": "No valid frames extracted. Video may be corrupted."}

    # Preprocess exactly as training: /255 → tensor → normalize (already 224x224)
    processed = []
    for img in frames:
        try:
            img = img / 255.0
            img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
            img = normalize(img)
            processed.append(img)
        except Exception:
            pass

    if len(processed) < num_frames:
        if len(processed) == 0:
            return {"error": "Frame preprocessing failed."}
        while len(processed) < num_frames:
            processed.append(processed[-1])

    x = torch.stack(processed).unsqueeze(0).to(device)
    x_flipped = x.flip(-1)  # TTA: horizontal flip

    try:
        with torch.no_grad():
            logits_orig, _ = model(x)
            logits_flip, _ = model(x_flipped)
            probs_orig = F.softmax(logits_orig, dim=1)[0]
            probs_flip = F.softmax(logits_flip, dim=1)[0]
            probs = (probs_orig + probs_flip) / 2.0

            # Heatmap activations
            outputs = model.model.videomae(pixel_values=x)
            hidden_states = outputs.last_hidden_state
            patch_activations = torch.norm(hidden_states, p=2, dim=-1)[0].cpu().numpy()
    except Exception as e:
        return {"error": f"Model inference failed: {str(e)}"}

    real_prob = probs[0].item()
    fake_prob = probs[1].item()

    # ── Forensics Boost (ELA + DCT) for subtle thin-layer deepfakes ──────────
    try:
        boosted_fake_prob, forensics_info = forensics_boost(frames, fake_prob)
    except Exception:
        boosted_fake_prob = fake_prob
        forensics_info    = {"ela_score": 0, "dct_score": 0, "ela_flag": False, "dct_flag": False}

    # Heatmap
    try:
        heatmaps = patch_activations.reshape(8, 14, 14)
        heatmaps = (heatmaps - heatmaps.min()) / (heatmaps.max() - heatmaps.min() + 1e-8)
        idx = min(4, len(frames) - 1)
        heatmap = heatmaps[idx // 2]
        heatmap_resized = cv2.resize(heatmap, (224, 224))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        original_frame = cv2.resize(frames[idx], (224, 224))
        overlay = cv2.addWeighted(original_frame, 0.6, heatmap_colored, 0.4, 0)
        _, buffer = cv2.imencode('.png', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        overlay_b64 = base64.b64encode(buffer).decode('utf-8')
        temporal_scores = heatmaps.mean(axis=(1, 2))
        frame_scores = np.repeat(temporal_scores, 2)
        frame_scores = (frame_scores / (frame_scores.max() + 1e-8)) * 100.0
    except Exception:
        overlay_b64 = ""
        frame_scores = [0.0] * 16

    file_size_mb = round(len(video_bytes) / (1024 * 1024), 2)

    return {
        "real_prob":        1.0 - fake_prob,
        "fake_prob":        fake_prob,
        "overlay_b64":      overlay_b64,
        "frame_scores":     frame_scores.tolist(),
        "frames_analyzed":  total_frames,
        "file_size_mb":     file_size_mb,
        "threshold_used":   FAKE_THRESHOLD,
        "suspect_threshold": SUSPECT_THRESHOLD,
    }


# ── API Endpoints ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return JSONResponse(content={"status": "ok", "device": str(device)})


@app.post("/predict")
async def predict(video: UploadFile = File(...)):
    contents = await video.read()
    # Run heavy CPU/GPU work in a thread pool so the event loop doesn't block
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(_executor, process_video_and_predict, contents)
    return JSONResponse(content=result)


class FeedbackRequest(BaseModel):
    filename: str
    model_prediction: str   # "REAL" or "FAKE"
    actual_label: str       # "REAL" or "FAKE" — user's correction
    fake_prob: float
    real_prob: float
    is_correct: bool


@app.post("/feedback")
async def submit_feedback(data: FeedbackRequest):
    """Save human feedback for later fine-tuning."""
    log = []
    if os.path.exists(FEEDBACK_LOG):
        try:
            with open(FEEDBACK_LOG, "r") as f:
                log = json.load(f)
        except Exception:
            log = []

    entry = data.dict()
    log.append(entry)

    with open(FEEDBACK_LOG, "w") as f:
        json.dump(log, f, indent=2)

    total = len(log)
    correct = sum(1 for e in log if e["is_correct"])
    return JSONResponse(content={
        "status": "saved",
        "total_feedback": total,
        "model_accuracy_so_far": round(correct / total * 100, 1) if total > 0 else 0,
    })


@app.get("/feedback/stats")
async def feedback_stats():
    """Return summary of all human feedback collected."""
    if not os.path.exists(FEEDBACK_LOG):
        return JSONResponse(content={"total": 0, "correct": 0, "accuracy": 0})
    with open(FEEDBACK_LOG, "r") as f:
        log = json.load(f)
    total = len(log)
    correct = sum(1 for e in log if e["is_correct"])
    wrong_fake_as_real = sum(1 for e in log if not e["is_correct"] and e["actual_label"] == "FAKE")
    wrong_real_as_fake = sum(1 for e in log if not e["is_correct"] and e["actual_label"] == "REAL")
    return JSONResponse(content={
        "total_feedback": total,
        "correct_predictions": correct,
        "accuracy_percent": round(correct / total * 100, 1) if total > 0 else 0,
        "missed_fakes": wrong_fake_as_real,
        "false_alarms": wrong_real_as_fake,
    })


# Serve frontend
from fastapi.staticfiles import StaticFiles
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
