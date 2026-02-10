# WASB-SBDT-FPFilter — Widely Applicable Strong Baseline for Sports Ball Detection and Tracking (English)

Repository for the paper: **[Widely Applicable Strong Baseline for Sports Ball Detection and Tracking](https://arxiv.org/abs/2311.05237)**

Shuhei Tarashima, Muhammad Abdul Haq, Yushan Wang, Norio Tagawa

[![arXiv](https://img.shields.io/badge/arXiv-2311.05237-00ff00.svg)](https://arxiv.org/abs/2311.05237) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

This English README is the canonical English-language documentation. A Chinese translation is available at [README_zh.md](README_zh.md).

## Overview
WASB-SBDT-FPFilter is a configurable baseline for sports ball detection & tracking across multiple sports (tennis, soccer, badminton, volleyball, basketball). This repository includes evaluation code, datasets, pretrained weights, and tools — plus a new end-to-end inference pipeline to automate detection, FP filtering and visualization.

Key components:
- Config-driven evaluation via Hydra (`src/main.py`)
- Multiple detectors (WASB, TrackNetV2, DeepBall, BallSeg variants)
- FP filter tooling (patch extraction, labeling, training, inference)
- End-to-end pipeline: `run_inference_pipeline.py`

---

## Quick Links
- Paper: https://arxiv.org/abs/2311.05237
- Pipeline docs: `PIPELINE_README.md`, `PIPELINE_使用说明.md` (Chinese), `PIPELINE_技术架构.md` (Chinese)
- Model weights: `MODEL_ZOO.md`
- English README: this file (`README_en.md`)
- 中文文档: `README_zh.md`

---

## Installation
Recommended: use Conda and Python 3.8+. Example:

```bash
conda create -n wasb-sbdt python=3.8 -y
conda activate wasb-sbdt
pip install -r requirements.txt
```

If `requirements.txt` is missing, install minimally:

```bash
pip install torch torchvision opencv-python pandas pillow tqdm omegaconf hydra-core matplotlib
```

For GPU acceleration, use CUDA-aware PyTorch binary built for your CUDA version.

---

## Download model weights
See `MODEL_ZOO.md` for download links and recommended weights. Place weights under `pretrained_weights/`.

---

## Dataset layout and CSV format
- Default dataset: `datasets/tennis_predict`
- Typical layout:
  - `datasets/<match>/<clip>/*.jpg` or `datasets/<match>_<clip>/*.jpg`
- Predictions CSV: `<match>_<clip>_predictions.csv` with columns:`file name`, `x-coordinate`, `y-coordinate`, `visibility`, `score`

---

## Running (Examples)

### 1) WASB evaluation (single-stage)

```bash
cd src
python main.py --config-name=eval dataset=tennis_predict model=wasb detector.model_path=../pretrained_weights/wasb_tennis_best.pth.tar runner.split=test runner.vis_result=True detector.step=3
```
- `detector.step`: 1 (every frame) or 3 (every 3 frames).

### 2) FP filter (standalone)

```bash
cd fp_filter
python inference.py --csv "../src/outputs/main/<timestamp>/match1_clip1_predictions.csv" --dataset-root "../datasets/tennis_predict" --model "patch_outputs/model_resnet/best.pth" --output "patch_outputs/patches_prediction/match1_clip1_predictions_filtered.csv" --threshold 0.5
```

### 3) Visualization (standalone)

```bash
cd fp_filter
python visualize_filtered.py --csv "../src/outputs/main/<timestamp>/match1_clip1_predictions.csv" --filtered-csv "patch_outputs/patches_prediction/match1_clip1_predictions_filtered.csv" --dataset-root "../src/outputs/main/<timestamp>" --output-video "patch_outputs/filtered_result.mp4" --fps 25
```

### 4) End-to-end Pipeline (recommended)

```bash
conda activate zsh
python run_inference_pipeline.py --dataset-root datasets/tennis_predict --wasb-weight pretrained_weights/wasb_tennis_best.pth.tar --fp-model fp_filter/patch_outputs/model_resnet/best.pth --output-base pipeline_outputs --step 3 --threshold 0.5 --fps 25
```
- Outputs stored under `<output-base>/<timestamp>/stage1_wasb_detection`, `stage2_fp_filtered`, `stage3_visualizations`.

---

## FP filtering workflow
1. Extract patches: `fp_filter/extract_patches.py` (generates `manifest.csv` and patch PNGs)
2. Label: `fp_filter/label_patches.py` (interactive labelling)
3. Train: `fp_filter/train_fp_filter.py`
4. Inference: `fp_filter/inference.py` (applies classifier to CSV)

---

## Configuration and Hydra
`src/main.py` uses Hydra. Override config values on CLI: e.g.

```bash
python main.py --config-name=eval dataset=tennis_predict detector.model_path=../pretrained_weights/wasb_tennis_best.pth.tar detector.step=1 runner.vis_result=True hydra.run.dir=outputs/myrun
```

Explore `src/configs/` for dataset, model, detector, runner, transform settings.

---

## Troubleshooting
- Numeric warnings like "Mean of empty slice": usually benign; check dataset annotations.
- CUDA OOM: reduce batch size or increase `--step`.
- Missing outputs: check logged stdout/stderr (pipeline records both).

---

## Contributing
Contributions welcome. Open issues/PRs and include tests where relevant.

---

## Citation
Please cite: https://arxiv.org/abs/2311.05237

---

*Switch to 中文: [README_zh.md](README_zh.md) *
