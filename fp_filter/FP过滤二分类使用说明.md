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

用标注好的 manifest 训练一个ResNet-18，输入为 patch 图像，输出二类：球(1) / 非球(0)。

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

## 数据说明

- **Patch 尺寸**：默认 96×96。可在第一步、第二步用同一 `--patch-size` 修改。
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

---

## 第五步：融合推理（ROI 内 WASB+FP，ROI 外 YOLO）

### 一键 Pipeline 已集成原图尺度标签生成

现在 `run_inference_pipeline.py` 已将 `fp_filter/csv_to_original_yolo.py` 纳入自动流程。

执行：

```powershell
python run_inference_pipeline.py ^
  --dataset-root datasets/tennis_predict ^
  --crop-left 650 --crop-top 51 ^
  --orig-w 1920 --orig-h 1080
```

运行完成后会在当前时间戳输出目录下新增：

- `stage5_original_yolo_labels/`
  - 每个 `*_predictions_filtered.csv` 对应一个子目录
  - 子目录命名规则：`<csv_stem>_orig_yolo_labels`
  - 目录中是可直接给 `hybrid_predict.py --wasb-labels-dir` 使用的原图尺度 YOLO 标签

如不希望 Stage 5 为空帧写空 txt，可增加参数：`--orig-no-save-empty`。

### 场景

你当前的目标是对 **1920×1080 原图**做最终推理，但在固定区域内（`left=650, top=51, right=1236, bottom=339`）优先采用 WASB+FP_Filter 的结果，其余区域采用 YOLO。

该方案可避免两个模型在 ROI 内重复产生冲突，同时保留 YOLO 在 ROI 外的覆盖能力。
此外，融合逻辑对 ROI 内的处理为“WASB 优先，YOLO 补位”：如果某帧 WASB/FP 已给出检测，则采用 WASB 结果；若某帧 WASB 无检测而 YOLO 在该帧 ROI 内有检测，则选取该帧 ROI 内 YOLO 置信度最高的检测作为补位。

### 推荐流程（稳定离线融合）

1. 先按既有流程跑出 FP 过滤结果 CSV：

- `*_predictions_filtered.csv`（坐标仍是裁剪图坐标系）

2. 将 FP 结果映射回原图 YOLO 坐标：

- 使用 `fp_filter/csv_to_original_yolo.py`
- 得到 `--wasb-labels-dir`（每帧一个 txt，原图归一化坐标）

3. 对原图运行融合脚本 `hybrid_predict.py`：

- ROI 内：优先使用 WASB+FP 框；若某帧 WASB 无检测且 YOLO 在该帧 ROI 内有检测，则从该帧 ROI 内的 YOLO 框中选取置信度最高的一个作为补位。
- ROI 外：保留 YOLO 框
- 最终输出统一 YOLO 标签

### 融合脚本

- 脚本位置：`hybrid_predict.py`（项目根目录）
- 主要参数：
  - `--input-folder`: 原图目录（1920×1080）
  - `--output-folder`: 融合结果 txt 输出目录
  - `--yolo-model`: YOLO 权重
  - `--wasb-labels-dir`: `csv_to_original_yolo.py` 生成的标签目录
  - `--left --top --right --bottom`: ROI 四点坐标
  - `--inside-policy`: ROI 归属规则（`center` 或 `any_overlap`）
  - `--nms-iou`: 融合后 NMS（0 表示关闭）

### 命令示例

```powershell
conda activate zsh

python hybrid_predict.py ^
  --input-folder datasets/tennis_predict/match1/clip1 ^
  --output-folder hybrid_outputs/match1_clip1_labels ^
  --yolo-model yolov8n_1280_1113.pt ^
  --wasb-labels-dir fp_filter/patch_outputs/patches_prediction/match1_clip1_orig_yolo_labels ^
  --conf 0.5 ^
  --orig-w 1920 --orig-h 1080 ^
  --left 650 --top 51 --right 1236 --bottom 339 ^
  --inside-policy center ^
  --nms-iou 0.0 ^
  --max-images 200
```

其中 `--max-images` 用于快速小样本验证；确认无误后可去掉该参数处理全量数据。

### 可靠性与健壮性设计说明

- **ROI 四点完整使用**：`left/top/right/bottom` 用于完整定义 ROI，而不是只使用偏移量。
- **越界与参数校验**：运行前校验 `left < right`、`top < bottom`、阈值范围是否合法。
- **容错读取标签**：WASB txt 某行格式异常时仅跳过该行并告警，不中断全流程。
- **可选冲突抑制**：可通过 `--nms-iou` 启用融合后按类别 NMS，减少边界重复框。
- **空帧一致性**：默认写空 txt，便于后续训练/评估流程保持帧对齐。

### 关键建议

- 若你希望“边界附近也强制交给 WASB”，可用 `--inside-policy any_overlap`。
- 若 ROI 外 YOLO 框与 ROI 内 WASB 框在边界处仍有重合，可设置 `--nms-iou 0.3`（可按效果调优）。
- 如果最终训练只关心单类别球，确保 YOLO 分支和 WASB 分支类别 ID 定义一致。

---

### CSV→YOLO 脚本选择与 `hybrid_predict.py` 可视化说明

- **脚本选择**：

  - 使用 `fp_filter/csv_to_original_yolo.py`（或 `run_inference_pipeline.py` 的 Stage5 输出）时，会把裁剪图坐标加上 `--crop-left` / `--crop-top` 的偏移并归一化到原图尺寸，生成 `*_orig_yolo_labels` 子目录，适用于 WASB+FP 在裁剪图场景下的离线融合流程。
  - 若仅需把 CSV 中点直接转换为 YOLO txt（无需坐标偏移），可使用 `fp_filter/csv_to_yolo_txt.py`。
- **hybrid_predict.py 的可视化**：`hybrid_predict.py` 默认会生成可视化图片与视频（前提：已安装 `opencv-python`）。如需关闭可使用 `--no-visualize` 或 `--no-visualize-video`；输出路径可通过 `--visualize-dir` / `--video-path` 指定，视频参数通过 `--video-fps` / `--video-fourcc` 控制。
