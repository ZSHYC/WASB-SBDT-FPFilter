# WASB-SBDT-FPFilter — 项目说明（中文）

论文链接：**[Widely Applicable Strong Baseline for Sports Ball Detection and Tracking](https://arxiv.org/abs/2311.05237)**

作者：Shuhei Tarashima, Muhammad Abdul Haq, Yushan Wang, Norio Tagawa

---

本中文版 README 为项目中文文档。英文版本在 `README_en.md`。

## 概述
WASB-SBDT-FPFilter 是用于多类体育项目（网球、足球、羽毛球、排球、篮球等）中小目标（球）检测与跟踪的基线实现。本仓库包含评估代码、示例数据、预训练权重、FP 过滤工具以及新增的一键推理 Pipeline（WASB 检测 → FP 过滤 → 可视化）。

主要功能：
- 基于 Hydra 的配置驱动评估（`src/main.py`）
- 支持多种检测模型和跟踪策略
- FP 过滤：patch 提取、标注、训练与推理工具
- 一键 Pipeline：`run_inference_pipeline.py`（自动串联三阶段）

---

## 快速链接
- 论文: https://arxiv.org/abs/2311.05237
- Pipeline 文档（中文）: `PIPELINE_使用说明.md`, `PIPELINE_技术架构.md`
- 模型权重: `MODEL_ZOO.md`
- 英文文档: `README_en.md`

---

## 安装
推荐使用 Conda：

```bash
conda create -n wasb-sbdt python=3.8 -y
conda activate wasb-sbdt
pip install -r requirements.txt
```

若无 `requirements.txt`，请安装基础依赖：

```bash
pip install torch torchvision opencv-python pandas pillow tqdm omegaconf hydra-core matplotlib
```

---

## 模型权重
参见 `MODEL_ZOO.md` 下载权重，放置在 `pretrained_weights/`，如：
```
pretrained_weights/wasb_tennis_best.pth.tar
```

---

## 数据格式
- 默认数据集：`datasets/tennis_predict`
- 目录结构：`datasets/<match>/<clip>/*.jpg` 或 `datasets/<match>_<clip>/*.jpg`
- 预测 CSV 格式：`<match>_<clip>_predictions.csv`，包含列：`file name`, `x-coordinate`, `y-coordinate`, `visibility`, `score`

---

## 常用运行示例

### 1) 单阶段 WASB 推理
```bash
cd src
python main.py --config-name=eval dataset=tennis_predict model=wasb detector.model_path=../pretrained_weights/wasb_tennis_best.pth.tar runner.split=test runner.vis_result=True detector.step=3
```

### 2) FP 过滤（单独运行）
```bash
cd fp_filter
python inference.py --csv "../src/outputs/main/<timestamp>/match1_clip1_predictions.csv" --dataset-root "../datasets/tennis_predict" --model "patch_outputs/model_resnet/best.pth" --output "patch_outputs/patches_prediction/match1_clip1_predictions_filtered.csv" --threshold 0.5
```

### 3) 可视化（单独运行）
```bash
cd fp_filter
python visualize_filtered.py --csv "../src/outputs/main/<timestamp>/match1_clip1_predictions.csv" --filtered-csv "patch_outputs/patches_prediction/match1_clip1_predictions_filtered.csv" --dataset-root "../src/outputs/main/<timestamp>" --output-video "patch_outputs/filtered_result.mp4" --fps 25
```

### 4) 一键全流程 Pipeline（推荐）
```bash
conda activate zsh
python run_inference_pipeline.py --dataset-root datasets/tennis_predict --wasb-weight pretrained_weights/wasb_tennis_best.pth.tar --fp-model fp_filter/patch_outputs/model_resnet/best.pth --output-base pipeline_outputs --step 3 --threshold 0.5 --fps 25
```

---

## FP 过滤流程（详细）
1. 从检测 CSV 提取 patch：`fp_filter/extract_patches.py`（生成 `manifest.csv` 与 patch 图像）
2. 标注 patch：`fp_filter/label_patches.py`（交互式标注）
3. 训练二分类模型：`fp_filter/train_fp_filter.py`
4. 运行过滤器：`fp_filter/inference.py`

---

## Hydra 与配置
`src/main.py` 使用 Hydra 配置系统。可在命令行覆盖配置项，例如：

```bash
python main.py --config-name=eval dataset=tennis_predict detector.model_path=../pretrained_weights/wasb_tennis_best.pth.tar detector.step=1 runner.vis_result=True hydra.run.dir=outputs/myrun
```

---

## 调试与常见问题
- 数值警告（如 `Mean of empty slice`）通常为统计上的告警；检查数据标注。
- CUDA OOM：降低 batch size 或增大 `--step`。
- 视频未生成：确认 Stage1 生成了可视化帧并在 Stage1 输出目录存在。
- 如果子命令返回成功但未生成文件，查看 pipeline 日志中的 stderr 输出。

---

## 贡献与测试
欢迎贡献：请提交 issue 或 PR 并附测试说明。建议在 CI 中添加轻量 smoke test 和 lint 检查。

---

## 引用
请引用论文：https://arxiv.org/abs/2311.05237

---

*Switch to English: [README_en.md](README_en.md)*
