"""
Translate timm-style checkpoint keys → HuggingFace VideoMAEForVideoClassification keys.
Run once: python fix_checkpoint.py
"""

import torch
import sys
import re

src = "outputs/checkpoints/best_model.pth"
dst = "outputs/checkpoints/best_model_hf.pth"

ckpt = torch.load(src, map_location="cpu")
print(f"Loaded checkpoint: {src}  ({len(ckpt)} keys)")

def timm_to_hf(k):
    # ── patch embedding ──────────────────────────────────────────────────────
    k = k.replace("backbone.patch_embed.proj.weight",
                  "model.videomae.embeddings.patch_embeddings.projection.weight")
    k = k.replace("backbone.patch_embed.proj.bias",
                  "model.videomae.embeddings.patch_embeddings.projection.bias")

    # ── positional / cls tokens ───────────────────────────────────────────────
    k = k.replace("backbone.cls_token",  "model.videomae.embeddings.cls_token")
    k = k.replace("backbone.pos_embed",  "model.videomae.embeddings.position_embeddings")

    # ── final norm ────────────────────────────────────────────────────────────
    k = k.replace("backbone.norm.weight", "model.videomae.layernorm.weight")
    k = k.replace("backbone.norm.bias",   "model.videomae.layernorm.bias")

    # ── transformer blocks ────────────────────────────────────────────────────
    m = re.match(r"backbone\.blocks\.(\d+)\.(.*)", k)
    if m:
        idx, rest = m.group(1), m.group(2)
        prefix = f"model.videomae.encoder.layer.{idx}."

        rest = rest.replace("norm1.weight",        "layernorm_before.weight")
        rest = rest.replace("norm1.bias",           "layernorm_before.bias")
        rest = rest.replace("norm2.weight",         "layernorm_after.weight")
        rest = rest.replace("norm2.bias",           "layernorm_after.bias")
        rest = rest.replace("attn.proj.weight",     "attention.output.dense.weight")
        rest = rest.replace("attn.proj.bias",       "attention.output.dense.bias")
        rest = rest.replace("attn.q_bias",          "attention.attention.q_bias")
        rest = rest.replace("attn.v_bias",          "attention.attention.v_bias")
        rest = rest.replace("attn.qkv.weight",      "attention.attention.qkv.weight")  # handle below
        rest = rest.replace("mlp.fc1.weight",       "intermediate.dense.weight")
        rest = rest.replace("mlp.fc1.bias",         "intermediate.dense.bias")
        rest = rest.replace("mlp.fc2.weight",       "output.dense.weight")
        rest = rest.replace("mlp.fc2.bias",         "output.dense.bias")
        k = prefix + rest

    # ── classifier head ───────────────────────────────────────────────────────
    k = k.replace("classifier.weight", "model.classifier.weight")
    k = k.replace("classifier.bias",   "model.classifier.bias")

    return k


# ── Build new state dict, splitting qkv if needed ────────────────────────────
new_ckpt = {}
for old_k, v in ckpt.items():
    if "attn.qkv.weight" in old_k:
        # Split Q K V into separate weight matrices
        m = re.match(r"backbone\.blocks\.(\d+)\.", old_k)
        if m:
            idx = m.group(1)
            pref = f"model.videomae.encoder.layer.{idx}.attention.attention."
            dim = v.shape[0] // 3
            new_ckpt[pref + "query.weight"] = v[:dim]
            new_ckpt[pref + "key.weight"]   = v[dim:2*dim]
            new_ckpt[pref + "value.weight"] = v[2*dim:]
        continue

    new_k = timm_to_hf(old_k)
    new_ckpt[new_k] = v

torch.save(new_ckpt, dst)
print(f"Saved translated checkpoint: {dst}  ({len(new_ckpt)} keys)")

# ── Quick validation ──────────────────────────────────────────────────────────
sys.path.insert(0, ".")
from src.models.videomae import VideoMAEClassifier
import torch.nn.functional as F

model = VideoMAEClassifier()
result = model.load_state_dict(new_ckpt, strict=False)
print(f"Missing  keys: {len(result.missing_keys)}")
print(f"Unexpected keys: {len(result.unexpected_keys)}")
if result.missing_keys:
    print("  Missing sample:", result.missing_keys[:5])
model.eval()

with torch.no_grad():
    dummy = torch.randn(1, 16, 3, 224, 224)
    logits, _ = model(dummy)
    probs = F.softmax(logits, dim=1)
    print(f"Random input probs (Real, Fake): {probs.tolist()}")

print("\n✅ Checkpoint translated and validated!")
