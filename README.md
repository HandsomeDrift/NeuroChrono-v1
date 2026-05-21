# NeuroChrono: Temporal Dynamics-Aware Brain-to-Video Reconstruction

[Drift](https://github.com/HandsomeDrift)

[![ArXiv](https://img.shields.io/badge/ArXiv-coming_soon-b31b1b.svg?logo=arXiv)](#)
[![Dataset](https://img.shields.io/badge/Dataset-CineBrain-faa035.svg?logo=Huggingface)](https://huggingface.co/datasets/Fudan-fMRI/CineBrain)

## Overview

Brain-to-video reconstruction aims to decode naturalistic visual perception from non-invasive neural recordings (fMRI, EEG) into video. **NeuroChrono** builds on the CineBrain-SF v2 dual-branch architecture and introduces **temporal dynamics-aware** conditioning — explicitly controlling *when* each neural guidance signal takes effect during the denoising process.

Three key innovations drive the current work:

1. **Timestep-Aware Alpha Scheduling** (Path A): Learned per-timestep modulation of multi-modal guidance signals, achieving **FVD 425** (−31% vs v2 baseline)
2. **Learnable Per-Sample Gating** (Path B): A gate network that predicts sample-specific α(τ) curves, achieving **FVD 440** with improved temporal consistency (EPE 3.01)
3. **OOD Temporal Asymmetry Theory**: Formal perturbation analysis proving that early denoising steps are **50.4×** more sensitive to condition perturbations than late steps — a fundamental property of the diffusion sampler with implications beyond brain-to-video

## Results

Within-subject evaluation on CineBrain dataset (sub-05, 540 test videos):

| Metric | CineBrain Baseline | SF v2 | **Path A** (E4_reverse) | **Path B** (P1 iter 500) |
|--------|:--:|:--:|:--:|:--:|
| **FVD** ↓ | 895.14 | 618.72 | **425.28** | **439.81** |
| **EPE** ↓ | 3.68 | 2.94 | 3.19 | **3.01** |
| SSIM ↑ | 0.288 | 0.302 | 0.282 | 0.279 |
| CLIP ↑ | 0.737 | 0.747 | 0.758 | 0.752 |
| Vid-50way ↑ | — | — | 0.372 | **0.377** |

- **Path A** (E4_reverse): Best FVD, static alpha schedule with reverse-sigmoid prior — motion guidance early, semantics late
- **Path B** (P1 iter 500): Near-Path-A FVD with better EPE and Vid-50way, learnable per-sample gating with prior bias

Cross-subject evaluation (train sub-05, test sub-03/04):

| Metric | CineBrain Baseline | SF v2 |
|--------|:--:|:--:|
| **FVD** ↓ | 936.65 | **684.06** (−27.0%) |
| **EPE** ↓ | 3.81 | **3.04** (−20.2%) |

## Architecture

### Base Architecture (from CineBrain-SF v2)

The foundation is a **Slow-Fast dual-branch** design inspired by the ventral-dorsal visual pathway:

- **Slow Branch (fMRI-driven)**: decodes semantic content, spatial structure, and keyframe priors — *what is in the video*
- **Fast Branch (EEG-driven)**: decodes motion dynamics, temporal changes, and scene transitions — *how the video changes*

The two branches are combined through **Cross-Modal Gated Fusion** with learned per-sample guidance weights, and injected into a CogVideoX-5B video diffusion model via a **Multi-Guidance Adapter**.

<p align="center">
  <img src="docs/figure/CineBrain-SF-v1-overview-v2.png" width="100%">
</p>

### Path A — Timestep-Aware Alpha Scheduling

Instead of static per-channel guidance weights, Path A modulates α(t) across denoising timesteps using an explicit schedule function:

- **Key finding**: Contrary to initial hypotheses, *late* brain-guidance amplification (reverse sigmoid) improves FVD while *early* amplification causes catastrophic degradation (FVD 1194 vs 425)
- **Asymmetry verified**: Early OOD (α_brain > 1.0 at τ=0) compounds across 49 steps → FVD disaster; same OOD at τ=1 affects only refinement → harmless

### Path B — Learnable Per-Sample Gating

Path B replaces the static schedule with a learned gate network:

```
α(τ) = gate_net(concat(pooled_features, timestep_embedding(τ))) + prior_bias(τ)
```

- The **prior bias** provides a safe floor (E4_reverse shape), guaranteeing FVD ≤ Path A
- The **gate network** learns to adjust α per-sample and per-timestep
- Path B iter 500 achieves Vid-50way 0.377 — the first time any variant surpasses Path A on video-level semantic recognition

### OOD Temporal Asymmetry Theory

A formal perturbation analysis of the diffusion sampling process reveals:

- **P1 (Monotonicity)**: Early-step condition perturbations produce **50.4×** larger output changes than late-step perturbations, as predicted by Theorem 6.2
- **H\*\* (Timing Asymmetry)**: The same OOD condition value (α_brain = 1.12) causes FVD collapse at τ=0 but is harmless at τ=1
- This asymmetry is a fundamental property of the DPM-Solver++ 2M sampler, not specific to brain signals

See `reference/THEORY_ood_asymmetry_v1.md` for full details.

## Method

### Slow Branch

The fMRI encoder (24-layer Transformer) processes visual and auditory ROI signals. Three prediction heads decode:
- **Keyframe Head**: predicts SigLIP image embeddings as visual priors
- **Scene-Text Head**: predicts text description embeddings for semantic guidance
- **Structure Head**: predicts VAE latent-space spatial layout

### Fast Branch

The EEG encoder (Conv1d + TCN + 12-layer Transformer) feeds into two pathways:
- **P0 Feature Distillation**: aligns EEG features to fMRI feature space via MSE distillation
- **P1 Temporal Dynamics Decoder**: extracts per-frame temporal change sequences from EEG

### Cross-Modal Gated Fusion

Separate projections for Slow/Fast features, combined via cross-attention mixing. A gating network learns per-sample weights for four guidance channels (keyframe, text, motion, brain latent). Path B extends this with timestep embedding input for temporal-aware gating.

### Multi-Guidance Adapter

Per-channel cross-attention injects guidance signals into the brain latent with spatial selectivity. Zero-initialized output projections ensure guidance grows from zero without disrupting the pretrained diffusion model.

### Training Pipeline

Three-stage progressive training with gradient isolation:

1. **Branch Pretrain**: train Slow and Fast branches independently with frozen DiT
2. **Fusion Training**: train Gated Fusion and Multi-Guidance Adapter with frozen branches
3. **Joint Finetuning**: end-to-end finetuning of all SF modules with LoRA on DiT

Path B adds a fourth phase: freeze Slow/Fast branches, unfreeze gate_net + fusion, train with prior bias and timestep embedding.

## Getting Started

This codebase is built on [CogVideoX-5B (SAT)](https://github.com/THUDM/CogVideo) with LoRA finetuning.

### Prerequisites

- GPU: NVIDIA A100/A800/H800 80GB (1 GPU for inference, 2-4 for training)
- Python 3.10+, PyTorch 2.6+, CUDA 12.4
- Dataset: [CineBrain](https://huggingface.co/datasets/Fudan-fMRI/CineBrain)

### Installation

```bash
git clone git@github.com:HandsomeDrift/NeuroChrono-v1.git
cd NeuroChrono-v1
pip install -r requirements.txt
```

For detailed data preparation, model weights, and path configuration, see [REPRODUCTION.md](REPRODUCTION.md).

### Inference

```bash
# v2 baseline inference
CUDA_VISIBLE_DEVICES=0 python sample_brain_va.py \
  --base configs/sf_v1/cinebrain_sf_v1_model.yaml configs/sf_v1/infer_stage3_v2.yaml \
  --seed 42 \
  --jsonpath /path/to/sub-0005_test_va.json \
  --output_dir results/sf_v2_sub05

# Path B inference (with learnable gating)
CUDA_VISIBLE_DEVICES=0 python sample_brain_va.py \
  --base configs/sf_v1/cinebrain_sf_v3_pathB_model.yaml configs/sf_v1/infer_pathB_p1_iter500.yaml \
  --seed 42 \
  --jsonpath /path/to/sub-0005_test_va.json \
  --output_dir results/pathB_iter500
```

### Training

```bash
# 4-GPU training (Path B example)
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
  --standalone --nproc_per_node=4 \
  train_video_fmri.py \
  --base configs/sf_v1/cinebrain_sf_v3_pathB_model.yaml configs/sf_v1/sf_v3_pathB_train.yaml \
  --seed 42
```

### Evaluation

```bash
python get_metric.py --sub 05
```

## Key Checkpoints

Important model weights are cataloged at `/public/home/maoyaoxin/zhangt/xxt/ckpts_archive/`:

| Symlink | Description | Key Metric |
|---------|-------------|------------|
| `03_stage3_v2_baseline` | SF v2 baseline | FVD 618.72 |
| `05_pathB_p1_main` | Path B P1 training (iter 500/1000/1500/2000) | FVD 439.81 |

See `ckpts_archive/README.md` for the complete catalog.

## Project History

NeuroChrono-v1 is the successor to **CineBrain-SF v3**. Full experimental history, design documents, and the OOD theory manuscript are archived at:

- Code archive: `/public/home/maoyaoxin/zhangt/xxt/SF-v3/`
- Project docs: `/home/drift/fitten/SF-v3/`

Key references in this repo:
- `reference/METHOD_PAPER_V2.md` — current paper draft (English)
- `reference/THEORY_ood_asymmetry_v1.md` — OOD temporal asymmetry theory
- `reference/SF-v3_HANDOFF_archive.md` — complete SF-v3 session history

## Citation

```bibtex
@article{neurochrono,
  title={NeuroChrono: Temporal Dynamics-Aware Brain-to-Video Reconstruction},
  author={Drift},
  year={2026}
}
```

This work builds upon CineBrain and CineBrain-SF:

```bibtex
@article{gao2025cinebrain,
  title={CineBrain: A Large-Scale Multi-Modal Brain Dataset During Naturalistic Audiovisual Narrative Processing},
  author={Gao, Jianxiong and Liu, Yichang and Yang, Baofeng and Feng, Jianfeng and Fu, Yanwei},
  journal={arXiv preprint arXiv:2503.06940},
  year={2025}
}
```

## Acknowledgements

- [CineBrain](https://github.com/yanweifu-sii/CineBrain) for the dataset and baseline codebase
- [CogVideoX](https://github.com/THUDM/CogVideo) for the video diffusion backbone
- [SAT](https://github.com/THUDM/SwissArmyTransformer) for the transformer training framework
