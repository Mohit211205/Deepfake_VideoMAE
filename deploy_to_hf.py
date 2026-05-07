"""
HF Spaces Upload Script
Uploads the entire project to HF Spaces using huggingface_hub Python API.
Run: conda run -n deepfake_env python deploy_to_hf.py
"""
import os
from huggingface_hub import HfApi, create_repo

# ─── CONFIG ───────────────────────────────────────────────────────────────────
HF_USERNAME = "AtrriJi"
SPACE_NAME = "deepfake-videomae"
SPACE_ID = f"{HF_USERNAME}/{SPACE_NAME}"
PROJECT_DIR = "."  # Run from project root
# ──────────────────────────────────────────────────────────────────────────────

api = HfApi()

# 1. Create the Space (Docker SDK)
print(f"\n[1/3] Creating HF Space: {SPACE_ID}")
try:
    create_repo(
        repo_id=SPACE_ID,
        repo_type="space",
        space_sdk="docker",
        exist_ok=True,
    )
    print(f"   ✅ Space ready: https://huggingface.co/spaces/{SPACE_ID}")
except Exception as e:
    print(f"   ❌ Error creating space: {e}")
    exit(1)

# 2. Define files to upload (exclude data/ and notebooks/)
SKIP_DIRS = {"data", "notebooks", ".git", "__pycache__", ".gradio", "outputs/visualizations"}
SKIP_FILES = {"debug.txt", "temp_video.mp4", "deploy_to_hf.py"}

files_to_upload = []
for root, dirs, files in os.walk(PROJECT_DIR):
    # Skip unwanted directories
    dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
    
    for file in files:
        filepath = os.path.join(root, file)
        rel_path = os.path.relpath(filepath, PROJECT_DIR)
        
        if any(rel_path.startswith(skip) for skip in SKIP_DIRS):
            continue
        if file in SKIP_FILES:
            continue
        if file.endswith(".pyc"):
            continue
            
        files_to_upload.append((filepath, rel_path))

print(f"\n[2/3] Uploading {len(files_to_upload)} files to HF Space...")
for i, (local_path, repo_path) in enumerate(files_to_upload):
    size_mb = os.path.getsize(local_path) / (1024 * 1024)
    print(f"   [{i+1}/{len(files_to_upload)}] {repo_path} ({size_mb:.1f} MB)")
    try:
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=SPACE_ID,
            repo_type="space",
        )
    except Exception as e:
        print(f"      ⚠️  Warning: {e}")

print(f"\n[3/3] ✅ DEPLOYMENT COMPLETE!")
print(f"   🌐 Your app: https://huggingface.co/spaces/{SPACE_ID}")
print(f"   ⏳ Build takes ~5-10 minutes. Check status at HF Spaces page.")
