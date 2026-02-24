# FP 过滤二分类：使用说明

通过“先提取 patch → 人工标注 → 训练二分类模型”的方式，筛除检测结果中的假阳性（FP），保留真阳性（TP）。

---

## 第一步：从检测结果提取 Patch

在完成 WASB 预测并得到 `match1_clip1_predictions.csv` 后，用脚本从**原始数据集**中，以每个 `visibility=1` 的检测点 (x, y) 为中心截取小图块（patch），并生成供标注用的 manifest。

### 1. 运行提取脚本

在 **`src`** 目录下执行（以下路径均相对 `src`）：

```powershell
cd C:\Users\mayuchao\Desktop\WASB-SBDT\src

python ../fp_filter/extract_patches.py ^
  "outputs/main/2026-02-05_11-10-50/match1_clip1_predictions.csv" ^
  --dataset-root ../datasets/tennis_predict ^
  --output-dir outputs/patches_match1_clip1 ^
  --patch-size 128
```

- **第一个参数**：预测结果 CSV 的路径。
- **`--dataset-root`**：原始数据集根目录（与 `configs/dataset/tennis_predict.yaml` 中的 `root_dir` 对应，即帧所在根目录）。
- **`--output-dir`**：输出目录，将在此生成所有 patch 图片和 `manifest.csv`。
- **`--patch-size`**：patch 边长（像素），默认 128（常见小目标尺寸）。若第一步用其他尺寸，第二步训练时需用 `--patch-size` 保持一致。

脚本会从 CSV 文件名解析 `match` 和 `clip`（如 `match1_clip1_predictions.csv` → match1, clip1），帧路径为：
`dataset_root / match / clip / 文件名`。
若目录结构不同，可用 `--match`、`--clip` 显式指定。

### 2. 输出内容

- **`output_dir/*.png`**：以 `(x,y)` 为中心截取的 patch 图。
- **`output_dir/manifest.csv`**：表格列包括
  `patch_id, source_file, x, y, match, clip, patch_path, score, label`。
  其中 **`label` 列为空**，供你标注。

### 3. 使用交互式标注脚本

使用 `fp_filter/label_patches.py` 工具，逐张查看图片并按键标注，效率极高。

在 **项目根目录** 下执行：

```powershell
python fp_filter/label_patches.py --manifest fp_filter/patch_outputs/patches_match1_clip1/manifest.csv --size 512
```

若需要从第 100 张图片开始标注（续接之前的进度）：

```powershell
python fp_filter/label_patches.py --manifest fp_filter/patch_outputs/patches_match1_clip1/manifest.csv --size 512 --start-from 100
```

**操作方法**：

- **按 `1`**：标记为 **球 (TP)**，自动保存并跳转下一张。
- **按 `0`**：标记为 **非球 (FP)**，自动保存并跳转下一张。
- **按 `B`**：回退到上一张（如果标错了）。
- **按 `Q`**：保存并退出。
- **UI 显示**：界面上会实时显示当前图片的标记状态（None / Ball(1) / Background(0)）。

---

## 第二步：训练二分类模型

用标注好的 manifest 训练一个小型 CNN，输入为 patch 图像，输出二类：球(1) / 非球(0)。

### 1. 准备

- 已完成第一步，且已在 `manifest.csv` 中填好 **`label`**（0 或 1）。
- `manifest.csv` 中的 `patch_path` 列指向的图片文件存在且可读。

### 2. 训练命令（在 `src` 下执行）

```powershell
cd C:\Users\mayuchao\Desktop\WASB-SBDT\src

python ../fp_filter/train_fp_filter.py ^
  --manifest outputs/patches_match1_clip1/manifest.csv ^
  --out-dir outputs/fp_filter ^
  --val-ratio 0.2 ^
  --epochs 50 ^
  --batch-size 64 ^
  --lr 1e-3 ^
  --patch-size 96
```

- **`--manifest`**：已标注 label 的 manifest 路径。
- **`--out-dir`**：保存 checkpoint 和 `history.json` 的目录。
- **`--patch-size`**：与第一步使用的 patch 尺寸一致（如 96）。

### 3. 输出

- **`outputs/fp_filter/best.pth`**：验证集准确率最高的模型，可用于后续推理筛 FP。
- **`outputs/fp_filter/last.pth`**：最后一轮模型。
- **`outputs/fp_filter/history.json`**：每轮 train/val 的 loss 与 accuracy。

## 模型与数据说明

- **Patch 尺寸**：默认 96×96。可在第一步、第二步用同一 `--patch-size` 修改。
- **二分类网络**：`tools/fp_filter/model.py` 中的轻量 CNN（若干 Conv+BN+ReLU+Pool，最后全连接 2 类）。如需更大/更小模型，可在此文件内修改或替换。
- **数据增强**：训练时仅做随机水平翻转；如需更多增强，可修改 `fp_filter/dataset.py` 中的 `get_default_transform`。

---

## 第三步：推理与过滤

使用训练好的 `best.pth` 对新的检测结果 CSV 进行过滤。

### 1. 运行推理脚本

在 **`src`** 目录下执行：

```powershell
cd C:\Users\mayuchao\Desktop\WASB-SBDT\src

python ../fp_filter/inference.py ^
  --csv "outputs/main/2026-02-05_11-10-50/match1_clip1_predictions.csv" ^
  --dataset-root ../datasets/tennis_predict ^
  --model "outputs/fp_filter/best.pth" ^
  --output "outputs/main/2026-02-05_11-10-50/match1_clip1_predictions_filtered.csv" ^
  --threshold 0.5
```

- **`--csv`**：待过滤的检测结果 CSV。
- **`--dataset-root`**：原始数据集根目录。
- **`--model`**：第二步训练好的模型路径。
- **`--output`**：输出的新 CSV 文件路径。
- **`--threshold`**：分类阈值（默认 0.5），低于此值的检测点将被视为 FP（`visibility` 置为 0）。

脚本会生成一个新的 CSV 文件，其中包含过滤后的结果，并增加 `fp_score` 列记录二分类模型的打分。

---

## 第四步：可视化过滤结果

为了更直观地评估 FP 过滤效果，仓库中提供了一个可视化脚本 `fp_filter/visualize_filtered.py`。脚本支持生成带标注的图片或合成对比视频，用于查看每一帧上哪些检测被保留、哪些被过滤。

### 左/右画面说明

- 如果以**对比视频或并排显示**方式查看（外部播放器或你另行实现）：
  - **左侧（Original / 原始检测）**：显示模型原始预测中的所有检测点（即推理 CSV 中的所有 visibility=1 条目），用于展示未经 FP 过滤前的检测分布。
  - **右侧（Filtered / 过滤后）**：显示经过 FP 过滤器处理后的结果（脚本会将低于阈值的检测标为不可见或移除），用于对比哪些点被认定为 FP 并被去掉。
- 我们提供的 `visualize_filtered.py` 默认并**不**生成真正的左右两图合成，而是在单张图上用不同颜色同时标注原始/过滤后的状态：
  - **绿色圆点**：保留的检测（TP，模型判断为球）。
  - **红色圆点**：被过滤的检测（FP，模型判断为非球）。
  - 可选 `--show-scores` 将在点旁显示 `fp_score`（保留概率）。

### 快速使用（图片）

在 `fp_filter` 下运行（示例）：

```powershell
python visualize_filtered.py \
  --csv "../src/outputs/main/2026-02-06_11-05-25/match1_clip1_predictions.csv" \
  --filtered-csv "patch_outputs/patches_prediction/match1_clip1_predictions_filtered.csv" \
  --dataset-root "../src/outputs/main/2026-02-06_11-05-25" \
  --output-dir "patch_outputs/visualizations" \
  --sample-rate 10 \
  --show-scores
```

### 生成对比视频（示例）

```powershell
python visualize_filtered.py \
  --csv "../src/outputs/main/2026-02-06_11-05-25/match1_clip1_predictions.csv" \
  --filtered-csv "patch_outputs/patches_prediction/match1_clip1_predictions_filtered.csv" \
  --dataset-root "../src/outputs/main/2026-02-06_11-05-25" \
  --output-video "patch_outputs/filtered_result.mp4" \
  --fps 25 \
  --show-scores
```

### 常见问题与注意事项

- 如果脚本提示找不到图片（"图像目录" 或 "图片不存在"），通常是因为 CSV 中的 `file name` 与 `dataset_root` 下的文件命名/目录结构不一致。常见两种目录格式：
  - `dataset_root/match/clip/<file.jpg>`（两层目录）
  - `dataset_root/match_clip`（单层目录，包含下划线）
    `visualize_filtered.py` 会尝试两者，但请确认 `--dataset-root` 指向正确的父目录。
- CSV 中若存在 `-inf`、`inf` 等非常值，会被跳过（脚本已自动过滤非有限坐标）。建议在运行前确认 CSV 中 `x-coordinate, y-coordinate` 的有效性。
- 若希望真正显示左右并排对比，可将两张图水平拼接后保存或在播放器中并列展示；`visualize_filtered.py` 可按需改为生成并排图。

---

## 附加：从推理 CSV 生成 YOLO txt 与可视化查看

为方便把 FP 过滤结果用于下游（如 YOLO 训练或人工快速检查），仓库里新增两个脚本：

- `fp_filter/csv_to_yolo_txt.py`：将推理后的 CSV（已过滤或未过滤）转换为逐帧的 YOLO 格式 `.txt`，一个帧一个 `.txt`。每行格式为 `class x_center y_center w h`（均为归一化到 [0,1] 的值）。脚本特点：
  - 支持自动从 `--image-root` 或 CSV 路径推断图像尺寸；若找不到图片，会基于 CSV 坐标估算一个后备尺寸（带下限以避免除零）。
  - 默认固定像素框大小为 15×15，可通过 `--box-size` 调整。
  - 可选择是否为无检测帧写入空 `.txt`（`--no-save-empty` 禁用）。

示例用法：

```powershell
conda activate zsh
python fp_filter/csv_to_yolo_txt.py \
  --csv fp_filter/patch_outputs/patches_prediction/match1_clip1_predictions_filtered.csv \
  --image-root datasets/tennis_predict \
  --box-size 15 \
  --class-id 0
```

运行后会在 CSV 同目录生成 `<csv_stem>_yolo_labels` 目录，里面每帧对应一个 `.txt`（默认同时会为无检测帧创建空文件，除非使用 `--no-save-empty`）。
