"""Quick A/B comparison of v2 vs v3_bugfix on 20-sample subset.

Computes FVD, EPE, SSIM, PSNR, CLIP Score, CTC on both result directories
using the same GT videos, so relative differences are meaningful even at n=20.
"""
import os
import sys
import time
import json
import numpy as np
import torch
import imageio.v3 as iio
from local_config import get_paths
from models.eval_metrics import (
    load_vit_model, load_clip_model,
    clip_score_only, ssim_score_only, psnr_score_only,
    clip_temporal_consistency, dino_temporal_consistency,
    compute_fvd, compute_epe,
)


def load_videos(result_dir, video_ids, n_frames=33):
    """Load output videos matching given IDs."""
    videos = []
    missing = []
    for vid in video_ids:
        p = os.path.join(result_dir, f"{str(vid).zfill(6)}.mp4")
        if not os.path.exists(p):
            missing.append(vid)
            continue
        arr = iio.imread(p)[:n_frames]
        videos.append(arr)
    return np.stack(videos) if videos else None, missing


def compute_metrics(pred, gt, device, tag=""):
    """Return dict of core metrics."""
    vit_proc, vit_mod = load_vit_model(device=device)
    clip_proc, clip_mod = load_clip_model(device=device)
    out = {}
    t = time.time()
    out["FVD"] = float(compute_fvd(pred, gt, device=device))
    print(f"  [{tag}] FVD: {out['FVD']:.4f} ({time.time()-t:.1f}s)")

    t = time.time()
    epe_mean, epe_std = compute_epe(pred, gt)
    out["EPE"] = float(epe_mean)
    print(f"  [{tag}] EPE: {epe_mean:.4f} +- {epe_std:.4f} ({time.time()-t:.1f}s)")

    t = time.time()
    ctc_mean, _ = clip_temporal_consistency(pred, device=device, preloaded=(clip_proc, clip_mod))
    out["CTC"] = float(ctc_mean)
    print(f"  [{tag}] CTC: {ctc_mean:.4f} ({time.time()-t:.1f}s)")

    # Per-frame SSIM/PSNR/CLIP (averaged)
    ssim_l, psnr_l, clip_l = [], [], []
    for fi in range(pred.shape[1]):
        s_mean, _ = ssim_score_only(pred[:, fi], gt[:, fi])
        p_mean, _ = psnr_score_only(pred[:, fi], gt[:, fi])
        c_mean, _ = clip_score_only(pred[:, fi], gt[:, fi], device=device,
                                     preloaded=(clip_proc, clip_mod))
        ssim_l.append(s_mean); psnr_l.append(p_mean); clip_l.append(c_mean)
    out["SSIM"] = float(np.mean(ssim_l))
    out["PSNR"] = float(np.mean(psnr_l))
    out["CLIP"] = float(np.mean(clip_l))
    print(f"  [{tag}] SSIM: {out['SSIM']:.4f} PSNR: {out['PSNR']:.2f} CLIP: {out['CLIP']:.4f}")
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = get_paths()

    v3_dir = sys.argv[1] if len(sys.argv) > 1 else "results/v3_bugfix_iter2500_mini20"
    v2_dir = sys.argv[2] if len(sys.argv) > 2 else "/public/home/maoyaoxin/zhangt/xxt/SF-v1/CineBrain/results/stage3_v2_sub05"
    subset_json = sys.argv[3] if len(sys.argv) > 3 else "/public/home/maoyaoxin/zhangt/xxt/datasets/sub-0005_test_va_mini20.json"

    with open(subset_json) as f:
        items = json.load(f)
    video_ids = sorted([int(os.path.basename(d["video"]).split(".")[0]) for d in items])
    print(f"Evaluating {len(video_ids)} videos: {video_ids[0]}-{video_ids[-1]}")

    # Load GT
    gt_videos, missing_gt = [], []
    for vid in video_ids:
        p = os.path.join(paths["video_dir"], f"{str(vid).zfill(6)}.mp4")
        if not os.path.exists(p):
            missing_gt.append(vid); continue
        gt_videos.append(iio.imread(p)[:33])
    gt = np.stack(gt_videos)
    print(f"GT shape: {gt.shape}, missing: {len(missing_gt)}")

    # Load v3_bugfix
    v3, miss_v3 = load_videos(v3_dir, video_ids)
    print(f"v3_bugfix shape: {v3.shape}, missing: {len(miss_v3)}")

    # Load v2
    v2, miss_v2 = load_videos(v2_dir, video_ids)
    print(f"v2 shape: {v2.shape}, missing: {len(miss_v2)}")

    # Metrics
    print("\n" + "=" * 60)
    print("  v2 baseline (iter 3000):")
    print("=" * 60)
    m_v2 = compute_metrics(v2, gt, device, "v2")

    print("\n" + "=" * 60)
    print("  v3_bugfix (iter 2500):")
    print("=" * 60)
    m_v3 = compute_metrics(v3, gt, device, "v3")

    # Delta
    print("\n" + "=" * 60)
    print("  Delta (v3_bugfix - v2):")
    print("=" * 60)
    for k in ["FVD", "EPE", "SSIM", "PSNR", "CLIP", "CTC"]:
        d = m_v3[k] - m_v2[k]
        arrow = "↓" if (k in ["FVD", "EPE"] and d < 0) or (k not in ["FVD", "EPE"] and d > 0) else "↑"
        sign = "+" if d >= 0 else ""
        tag = "BETTER" if (k in ["FVD", "EPE"] and d < 0) or (k not in ["FVD", "EPE"] and d > 0) else "WORSE"
        print(f"  {k:8s}  v2={m_v2[k]:.4f}  v3={m_v3[k]:.4f}  Δ={sign}{d:.4f}  [{tag}]")


if __name__ == "__main__":
    main()
