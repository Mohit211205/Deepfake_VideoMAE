"""
Reorganize DINOv2 frames into VideoMAE folder structure.
DINOv2: faces/{real,fake}/videoname_frame_XXXX.jpg  (flat)
VideoMAE needs: faces/{real,fake}/video_folder/frame_XXX.jpg

Creates symlinks (no file copying, instant).
Run: python reorganize_dino_dataset.py
"""
import os
import re
from pathlib import Path
from collections import defaultdict

DINO_FACES = "/usershome/cs671_user16/projectsG50/Deepfake_DINOv2_01/data/faces"
OUT_FACES  = "/usershome/cs671_user16/projectsG50/Deepfake_VideoMAE_DomainSSL/data/faces_dino"

def reorganize(label):
    src_dir = Path(DINO_FACES) / label
    out_dir = Path(OUT_FACES) / label
    out_dir.mkdir(parents=True, exist_ok=True)

    # Group frames by video prefix (everything before _frame_XXXX)
    video_groups = defaultdict(list)
    for jpg in sorted(src_dir.glob("*.jpg")):
        # e.g. 01_02__exit_phone_room__YVGY8LOK_frame_0042.jpg
        m = re.match(r"^(.+)_frame_(\d+)\.jpg$", jpg.name)
        if m:
            video_id = m.group(1)
            video_groups[video_id].append(jpg)
        else:
            # No frame pattern — treat entire image as single frame
            video_groups[jpg.stem].append(jpg)

    print(f"\n  {label}: {len(video_groups)} videos, "
          f"{sum(len(v) for v in video_groups.values())} total frames")

    created = 0
    for video_id, frames in video_groups.items():
        vfolder = out_dir / video_id
        if vfolder.exists() and len(list(vfolder.iterdir())) >= min(len(frames), 4):
            continue  # already done
        vfolder.mkdir(exist_ok=True)
        for i, src_jpg in enumerate(sorted(frames)):
            dst = vfolder / f"frame_{i:03d}.jpg"
            if not dst.exists():
                dst.symlink_to(src_jpg.resolve())
        created += 1

    print(f"  ✅ {created} new video folders created in {out_dir}")
    return len(video_groups)

if __name__ == "__main__":
    print("="*60)
    print("  Reorganizing DINOv2 dataset for VideoMAE")
    print("="*60)
    n_real = reorganize("real")
    n_fake = reorganize("fake")
    print(f"\n📊 Final dataset:")
    print(f"   REAL : {n_real} videos")
    print(f"   FAKE : {n_fake} videos")
    print(f"   TOTAL: {n_real + n_fake} videos")
    print(f"\n  Output: {OUT_FACES}")
    print(f"\n  Update train.py: dataset = DeepfakeDataset('{OUT_FACES}', ...)")
    print("="*60)
