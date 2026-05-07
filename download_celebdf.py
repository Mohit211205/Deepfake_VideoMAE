"""
Celeb-DF Dataset Downloader + Face Extractor
Downloads Celeb-DF v2 from Google Drive and extracts faces into data/faces/
Run from project root: python download_celebdf.py
"""
import os
import subprocess
import cv2
import gdown
from pathlib import Path
from facenet_pytorch import MTCNN
from PIL import Image
import torch

# ── Config ────────────────────────────────────────────────────
OUTPUT_DIR = "data/faces"
TEMP_DIR = "data/celebdf_raw"
NUM_FRAMES_PER_VIDEO = 16
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")  # GPU 7 has 38GB free

# Celeb-DF v2 Google Drive file IDs
# Source: https://github.com/yuezunli/celeb-deepfakeforensics
CELEB_DF_FILES = {
    # Real videos
    "Celeb-real.zip": "1D3YkNqi4Klu9SXwSHkDaEhPHjHEhHFHl",
    # Fake videos (celebrity deepfakes)
    "Celeb-synthesis.zip": "1Tz_3W_S09tFkuLRQSm-dOZrQ1kX3QQYP",
    # YouTube real videos (extra real data)
    "YouTube-real.zip": "1iLx76wsbi9itnkxSqz9BVBl4ZvnbIazj",
}

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/real", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/fake", exist_ok=True)

mtcnn = MTCNN(keep_all=False, device=device)

# ── Step 1: Download ──────────────────────────────────────────
print("="*60)
print("STEP 1: Downloading Celeb-DF v2")
print("="*60)

for filename, gdrive_id in CELEB_DF_FILES.items():
    output_path = os.path.join(TEMP_DIR, filename)
    if os.path.exists(output_path):
        print(f"  ✅ {filename} already downloaded, skipping.")
        continue
    print(f"\n  Downloading {filename}...")
    gdown.download(id=gdrive_id, output=output_path, quiet=False)
    print(f"  ✅ {filename} downloaded!")

# ── Step 2: Extract Zips ──────────────────────────────────────
print("\n" + "="*60)
print("STEP 2: Extracting zips")
print("="*60)

for filename in CELEB_DF_FILES.keys():
    zip_path = os.path.join(TEMP_DIR, filename)
    extract_dir = os.path.join(TEMP_DIR, filename.replace(".zip", ""))
    if os.path.exists(extract_dir):
        print(f"  ✅ {filename} already extracted, skipping.")
        continue
    if os.path.exists(zip_path):
        print(f"  Extracting {filename}...")
        subprocess.run(["unzip", "-q", zip_path, "-d", TEMP_DIR], check=True)
        print(f"  ✅ Extracted!")

# ── Step 3: Face Extraction Function ─────────────────────────
def extract_faces_from_video(video_path, out_folder, label, video_idx):
    """Extract NUM_FRAMES_PER_VIDEO face crops from a video."""
    folder_name = f"celebdf_{label}_{video_idx:05d}"
    out_path = os.path.join(out_folder, folder_name)

    if os.path.exists(out_path) and len(os.listdir(out_path)) >= NUM_FRAMES_PER_VIDEO:
        return True  # Already done

    os.makedirs(out_path, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return False

    skip = max(1, total // NUM_FRAMES_PER_VIDEO)
    frames_saved = 0
    count = 0

    while frames_saved < NUM_FRAMES_PER_VIDEO:
        ret, frame = cap.read()
        if not ret:
            break
        if count % skip == 0:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)
                boxes, _ = mtcnn.detect(pil)
                if boxes is not None and len(boxes) > 0:
                    b = boxes[0]
                    x1, y1, x2, y2 = [int(v) for v in b]
                    w, h = x2-x1, y2-y1
                    x1 = max(0, x1 - int(w*0.2))
                    y1 = max(0, y1 - int(h*0.2))
                    x2 = min(rgb.shape[1], x2 + int(w*0.2))
                    y2 = min(rgb.shape[0], y2 + int(h*0.2))
                    crop = rgb[y1:y2, x1:x2]
                else:
                    h, w = rgb.shape[:2]
                    m = min(h, w)
                    crop = rgb[(h-m)//2:(h+m)//2, (w-m)//2:(w+m)//2]

                crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(out_path, f"frame_{frames_saved:03d}.jpg"), crop_bgr)
                frames_saved += 1
            except Exception:
                pass
        count += 1
    cap.release()
    return frames_saved > 0

# ── Step 4: Process Videos ────────────────────────────────────
print("\n" + "="*60)
print("STEP 3: Extracting faces from videos")
print("="*60)

# Map source folders to labels
sources = [
    (f"{TEMP_DIR}/Celeb-real",       "real"),
    (f"{TEMP_DIR}/YouTube-real",     "real"),
    (f"{TEMP_DIR}/Celeb-synthesis",  "fake"),
]

real_count, fake_count = 0, 0
for src_dir, label in sources:
    if not os.path.exists(src_dir):
        print(f"  ⚠️  {src_dir} not found, skipping.")
        continue

    videos = list(Path(src_dir).rglob("*.mp4")) + list(Path(src_dir).rglob("*.avi"))
    out_folder = f"{OUTPUT_DIR}/{label}"
    counter = real_count if label == "real" else fake_count

    print(f"\n  Processing {len(videos)} {label} videos from {src_dir}")
    for i, vpath in enumerate(videos):
        ok = extract_faces_from_video(vpath, out_folder, label, counter + i)
        if ok:
            if label == "real": real_count += 1
            else: fake_count += 1
        if (i+1) % 50 == 0:
            print(f"    [{i+1}/{len(videos)}] {label} — {real_count} real, {fake_count} fake extracted so far")

print("\n" + "="*60)
print(f"✅ DONE! Dataset Summary:")
print(f"   Real videos: {real_count}")
print(f"   Fake videos: {fake_count}")
print(f"   Total: {real_count + fake_count}")
print(f"\nYour existing data/faces/real and data/faces/fake now have Celeb-DF added!")
print("="*60)
