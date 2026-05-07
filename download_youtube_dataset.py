"""
YouTube Deepfake Dataset Builder
Downloads diverse fake + real face videos from YouTube and extracts faces.
Run: PYTHONPATH=. python download_youtube_dataset.py
"""
import os
import subprocess
import cv2
import torch
from PIL import Image
from pathlib import Path
from facenet_pytorch import MTCNN

# ── Config ────────────────────────────────────────────────────
OUTPUT_DIR      = "data/faces"
TEMP_DIR        = "data/youtube_raw"
NUM_FRAMES      = 16
MAX_DURATION    = 60          # seconds — skip very long videos
FAKE_LIMIT      = 300         # how many new fake videos to add
REAL_LIMIT      = 300         # how many new real videos to add
CUDA_DEVICE     = "cuda:7"    # GPU 7 — most free

os.makedirs(f"{OUTPUT_DIR}/fake", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/real", exist_ok=True)
os.makedirs(f"{TEMP_DIR}/fake", exist_ok=True)
os.makedirs(f"{TEMP_DIR}/real", exist_ok=True)

device = torch.device(CUDA_DEVICE if torch.cuda.is_available() else "cpu")
mtcnn  = MTCNN(keep_all=False, device=device)

# ── YouTube Sources ───────────────────────────────────────────
# Curated: Each URL is a PUBLIC playlist or individual video
# that is well-known to contain deepfakes/real faces.

FAKE_SOURCES = [
    # Deeptomcruise deepfake examples (very obvious, well labeled)
    "https://www.youtube.com/playlist?list=PLo4vh38_F3kfGFzgUU9DLjb-MxIV5Gk_L",
    # Various deepfake detection research demo videos
    "https://www.youtube.com/watch?v=cQ54GDm1eL0",
    "https://www.youtube.com/watch?v=dMF2i3A9Lzw",
    "https://www.youtube.com/watch?v=OCLaeBAkFAY",
    "https://www.youtube.com/watch?v=mUfJOQKdtAk",
    "https://www.youtube.com/watch?v=VWrhRBb-1Ig",
    "https://www.youtube.com/watch?v=Amy8znrCjJk",
    "https://www.youtube.com/watch?v=9Yq67CjDqvw",
    "https://www.youtube.com/watch?v=p_8uu0Y_sLs",
    "https://www.youtube.com/watch?v=Cg6_TUVNLTQ",
    "https://www.youtube.com/watch?v=GLoI9xovDYk",
]

REAL_SOURCES = [
    # Celebrity interviews (clear real face footage)
    "https://www.youtube.com/watch?v=V-_O7nl0Ii0",   # Obama speech
    "https://www.youtube.com/watch?v=jrTgkKDwWNs",   # Elon interview
    "https://www.youtube.com/watch?v=uMK0prafzw0",   # News anchor
    "https://www.youtube.com/watch?v=pVEeXjPiw54",   # TED talk
    "https://www.youtube.com/watch?v=Unzc731iCUY",   # MIT lecture
    "https://www.youtube.com/watch?v=6wXkI4t7nuc",   # Talk show
    "https://www.youtube.com/watch?v=arj7oStGLkU",   # News clip
    "https://www.youtube.com/watch?v=ZtCeRPiMPYI",   # Interview
    "https://www.youtube.com/watch?v=eC7GHMmnCRQ",   # Lecture
    "https://www.youtube.com/watch?v=qbW6FRbaSl0",   # News
]

# ── Download Function ─────────────────────────────────────────
def download_videos(urls, out_dir, label, limit):
    print(f"\n{'='*55}")
    print(f"  Downloading {label.upper()} videos (max {limit})")
    print(f"{'='*55}")

    ydl_opts = [
        "yt-dlp",
        "--format", "mp4[height<=480]/best[height<=480]/best",
        "--output", f"{out_dir}/%(id)s.%(ext)s",
        "--max-downloads", str(limit),
        "--match-filter", f"duration < {MAX_DURATION}",
        "--no-playlist" if not any("playlist" in u for u in urls) else "--yes-playlist",
        "--ignore-errors",
        "--no-warnings",
        "--quiet",
        "--progress",
    ]

    for url in urls:
        print(f"  Fetching: {url[:60]}...")
        try:
            result = subprocess.run(
                ydl_opts + [url],
                capture_output=False, timeout=300
            )
        except subprocess.TimeoutExpired:
            print(f"  ⏱️ Timeout on {url}")
        except Exception as e:
            print(f"  ❌ Error: {e}")

    downloaded = list(Path(out_dir).glob("*.mp4")) + list(Path(out_dir).glob("*.webm"))
    print(f"  ✅ {len(downloaded)} {label} videos downloaded.")
    return downloaded

# ── Face Extraction ───────────────────────────────────────────
def extract_faces(video_path, out_base, label, idx):
    folder_name = f"yt_{label}_{idx:05d}"
    out_path    = os.path.join(out_base, folder_name)

    if os.path.exists(out_path) and len(os.listdir(out_path)) >= NUM_FRAMES:
        return True

    os.makedirs(out_path, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return False

    skip  = max(1, total // NUM_FRAMES)
    saved = 0
    count = 0

    while saved < NUM_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        if count % skip == 0:
            try:
                rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil  = Image.fromarray(rgb)
                boxes, _ = mtcnn.detect(pil)
                if boxes is not None and len(boxes) > 0:
                    b            = boxes[0]
                    x1,y1,x2,y2 = [int(v) for v in b]
                    w, h         = x2-x1, y2-y1
                    x1 = max(0, x1-int(w*0.2));  y1 = max(0, y1-int(h*0.2))
                    x2 = min(rgb.shape[1], x2+int(w*0.2))
                    y2 = min(rgb.shape[0], y2+int(h*0.2))
                    crop = rgb[y1:y2, x1:x2]
                else:
                    h2,w2 = rgb.shape[:2];  m = min(h2,w2)
                    crop  = rgb[(h2-m)//2:(h2+m)//2, (w2-m)//2:(w2+m)//2]
                if crop.size == 0:
                    count += 1; continue
                cv2.imwrite(
                    os.path.join(out_path, f"frame_{saved:03d}.jpg"),
                    cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                )
                saved += 1
            except Exception:
                pass
        count += 1
    cap.release()
    return saved >= 4  # Accept if at least 4 frames extracted

# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    # Step 1: Download
    fake_videos = download_videos(FAKE_SOURCES, f"{TEMP_DIR}/fake", "fake", FAKE_LIMIT)
    real_videos = download_videos(REAL_SOURCES, f"{TEMP_DIR}/real", "real", REAL_LIMIT)

    # Step 2: Extract faces
    print(f"\n{'='*55}")
    print("  Extracting faces...")
    print(f"{'='*55}")

    fake_ok = 0
    for i, vp in enumerate(fake_videos):
        ok = extract_faces(vp, f"{OUTPUT_DIR}/fake", "fake", i)
        if ok: fake_ok += 1
        if (i+1) % 20 == 0:
            print(f"  Fake: {i+1}/{len(fake_videos)} processed, {fake_ok} extracted")

    real_ok = 0
    for i, vp in enumerate(real_videos):
        ok = extract_faces(vp, f"{OUTPUT_DIR}/real", "real", i)
        if ok: real_ok += 1
        if (i+1) % 20 == 0:
            print(f"  Real: {i+1}/{len(real_videos)} processed, {real_ok} extracted")

    print(f"\n{'='*55}")
    print(f"✅ Dataset expanded!")
    print(f"   New fake videos added : {fake_ok}")
    print(f"   New real videos added : {real_ok}")

    # Count totals
    total_fake = len(os.listdir(f"{OUTPUT_DIR}/fake"))
    total_real = len(os.listdir(f"{OUTPUT_DIR}/real"))
    print(f"\n📊 Total dataset now:")
    print(f"   REAL: {total_real}  |  FAKE: {total_fake}  |  TOTAL: {total_real+total_fake}")
    print(f"\nNow run: PYTHONPATH=. python src/train.py")
    print(f"{'='*55}")
