# 完整推理 Pipeline 使用说明

## 概述

`run_inference_pipeline.py` 是一个自动化脚本，将网球检测系统的四个核心步骤串联成一个完整的 Pipeline：

1. **Stage 1: WASB 球检测** - 使用 WASB 模型对新数据进行初步检测
2. **Stage 2: FP 误检过滤** - 使用训练好的二分类模型剔除假阳性检测
3. **Stage 3: 结果可视化** - 生成包含原始检测和过滤后结果的对比视频
4. **Stage 4: YOLO 标签生成** - 将过滤后的 CSV 转换为逐帧 YOLO txt 标签文件

通过一行命令即可完成从原始图片到最终检测视频与 YOLO 标签的全流程处理。

---

## 前置条件

### 1. 数据准备

确保新的预测数据集已放置在正确位置：

```
datasets/tennis_predict/
└── match1/
    └── clip1/
        ├── 0001.jpg
        ├── 0002.jpg
        ├── 0003.jpg
        └── ...
```

**注意事项：**

- 图片文件名格式：`0001.jpg`, `0002.jpg` 等（4位数字+.jpg）
- 必须包含在 `match/clip` 两层目录结构中
- 如果有多个 match 或 clip，可以按同样结构添加

### 2. 模型权重文件

确认以下模型文件已存在：

- **WASB 检测模型**: `pretrained_weights/wasb_tennis_best.pth.tar`
- **FP 过滤模型**: `fp_filter/patch_outputs/model_resnet/best.pth`

### 3. Python 环境

确保已安装所有依赖包（参考项目 `README.md`）：

```bash
pip install torch torchvision opencv-python pandas pillow tqdm omegaconf hydra-core
```

---

## 快速开始

### 最简单用法（使用默认参数）

在项目根目录下执行：

```bash
python run_inference_pipeline.py
```

这将使用默认配置处理 `datasets/tennis_predict` 下的所有数据。

### 查看帮助信息

```bash
python run_inference_pipeline.py --help
```

---

## 参数说明

### 输入路径参数

| 参数               | 默认值                                            | 说明              |
| ------------------ | ------------------------------------------------- | ----------------- |
| `--dataset-root` | `datasets/tennis_predict`                       | 原始数据集根目录  |
| `--wasb-weight`  | `pretrained_weights/wasb_tennis_best.pth.tar`   | WASB 模型权重文件 |
| `--fp-model`     | `fp_filter/patch_outputs/model_resnet/best.pth` | FP 过滤器模型权重 |

### 输出路径参数

| 参数              | 默认值               | 说明                                            |
| ----------------- | -------------------- | ----------------------------------------------- |
| `--output-base` | `pipeline_outputs` | Pipeline 输出基础目录（会自动添加时间戳子目录） |

### 推理参数

| 参数            | 默认值  | 说明                                                                   |
| --------------- | ------- | ---------------------------------------------------------------------- |
| `--step`      | `1`   | WASB 检测步长。`1`=逐帧检测（精确但慢），`3`=每3帧检测一次（快速） |
| `--threshold` | `0.5` | FP 过滤器阈值（0-1）。越高越严格，会过滤掉更多检测点                   |
| `--fps`       | `25`  | 输出视频帧率                                                           |
| `--box-size`  | `15`  | Stage 4 YOLO 标签中固定框的像素边长                                    |
| `--class-id`  | `0`   | Stage 4 YOLO 标签的类别 ID                                             |

---

## 使用示例

### 示例 1: 使用默认配置

```bash
python run_inference_pipeline.py
```

### 示例 2: 高精度模式（逐帧检测）

```bash
python run_inference_pipeline.py --step 1
```

### 示例 3: 调整 FP 过滤阈值

如果发现结果中仍有较多误检，可以提高阈值：

```bash
python run_inference_pipeline.py --threshold 0.7
```

如果发现过滤太严格（漏检真球），可以降低阈值：

```bash
python run_inference_pipeline.py --threshold 0.3
```

### 示例 4: 处理不同的数据集

```bash
python run_inference_pipeline.py ^
    --dataset-root "datasets/my_custom_dataset" ^
    --output-base "my_results"
```

### 示例 5: 完整参数示例

```bash
python run_inference_pipeline.py ^
    --dataset-root "datasets/tennis_predict" ^
    --wasb-weight "pretrained_weights/wasb_tennis_best.pth.tar" ^
    --fp-model "fp_filter/patch_outputs/model_resnet/best.pth" ^
    --output-base "pipeline_outputs" ^
    --step 1 ^
    --threshold 0.5 ^
    --fps 25 ^
    --box-size 15 ^
    --class-id 0
```

---

## 输出结果说明

### 输出目录结构

运行后会在 `pipeline_outputs` 下创建带时间戳的目录：

```
pipeline_outputs/
└── 2026-02-10_15-30-45/    ← 本次运行的时间戳
    ├── stage1_wasb_detection/
    │   ├── match1_clip1_predictions.csv       ← Stage 1: 原始检测结果
    │   ├── match1_clip1/                      ← Stage 1: 可视化图片序列
    │   │   ├── 0001.jpg
    │   │   ├── 0002.jpg
    │   │   └── ...
    │   └── .hydra/                            ← Hydra 配置信息
    │
    ├── stage2_fp_filtered/
    │   └── match1_clip1_predictions_filtered.csv  ← Stage 2: 过滤后的检测结果
    │
    ├── stage3_visualizations/
    │   └── match1_clip1_final_result.mp4      ← Stage 3: 最终对比视频
    │
    └── stage4_yolo_labels/
        └── match1_clip1_predictions_filtered_yolo_labels/
            ├── 00000001.txt                   ← Stage 4: 每帧对应的 YOLO txt
            ├── 00000002.txt
            └── ...
```

### 各阶段输出说明

#### Stage 1: WASB 检测结果

- **CSV 文件** (`*_predictions.csv`): 包含每帧的检测结果

  - `file name`: 图片文件名
  - `x-coordinate`: 检测到的球的 x 坐标
  - `y-coordinate`: 检测到的球的 y 坐标
  - `visibility`: 可见性（0=不可见，1=可见）
  - `score`: 检测置信度分数
- **可视化图片**: 每张图片显示左侧为真实标注（如果有），右侧为检测结果

#### Stage 2: FP 过滤结果

- **Filtered CSV** (`*_predictions_filtered.csv`): 过滤后的检测结果
  - 新增 `fp_score` 列：二分类模型预测的"假阳性概率"
  - 新增 `fp_pred` 列：判定结果（0=真球，1=误检）
  - `visibility` 列：误检点的 visibility 已被置为 0

#### Stage 3: 可视化视频

- **MP4 视频**: 对比展示原始检测和过滤后的结果
  - 🔴 红色圆圈：原始 WASB 检测结果
  - 🟢 绿色圆圈：FP 过滤后保留的结果
  - 可以直观看到哪些误检被成功过滤掉

#### Stage 4: YOLO 标签

- **逐帧 txt 文件**（存放于 `stage4_yolo_labels/<stem>_yolo_labels/`）
  - 每帧对应一个 `.txt`，无检测时文件为空
  - 每行格式：`class x_center y_center width height`（均为 [0,1] 归一化值）
  - 框大小固定为 `--box-size` 指定的像素值（默认 15×15），可按需调整
  - 可直接用于 YOLO 系列模型的训练或评估数据集

---

## 常见问题排查

### 问题 1: 找不到数据集

```
FileNotFoundError: 数据集目录不存在: datasets/tennis_predict
```

**解决方案**:

- 检查数据集路径是否正确
- 使用 `--dataset-root` 参数指定正确路径

### 问题 2: 找不到模型权重

```
FileNotFoundError: WASB 权重文件不存在: pretrained_weights/wasb_tennis_best.pth.tar
```

**解决方案**:

- 确认模型文件已下载到正确位置
- 参考 `MODEL_ZOO.md` 下载模型权重

### 问题 3: Stage 1 未生成 CSV 文件

```
未找到生成的 predictions.csv 文件
```

**可能原因**:

- 数据集目录结构不符合要求（必须是 `match/clip/图片` 结构）
- `tennis_predict.yaml` 配置文件设置不正确

**解决方案**:

- 检查 `src/configs/dataset/tennis_predict.yaml`
- 确保 `test.matches` 设置为 `'all'` 或包含你的 match 名称

### 问题 4: CUDA out of memory

```
RuntimeError: CUDA out of memory
```

**解决方案**:

- 使用更大的 `--step` 值（如 `--step 5`）减少批量大小
- 或者在代码中修改 batch_size（需要修改配置文件）

### 问题 5: 视频生成失败

```
视频文件未生成
```

**可能原因**:

- Stage 1 的可视化图片未正确生成
- OpenCV 视频编码器问题

**解决方案**:

- 检查 Stage 1 输出目录中是否有图片文件
- 尝试安装 `opencv-python-headless` 或更新 OpenCV

---

## 高级用法

### 1. 批量处理多个数据集

创建一个批处理脚本 `batch_process.bat`:

```batch
@echo off
python run_inference_pipeline.py --dataset-root "datasets/match1"
python run_inference_pipeline.py --dataset-root "datasets/match2"
python run_inference_pipeline.py --dataset-root "datasets/match3"
```

### 2. 只运行部分阶段

如果只想重新运行某个阶段，可以直接调用对应脚本：

**重新运行 Stage 2 (FP 过滤)**:

```bash
cd fp_filter
python inference.py ^
    --csv "../pipeline_outputs/2026-02-10_15-30-45/stage1_wasb_detection/match1_clip1_predictions.csv" ^
    --dataset-root "../datasets/tennis_predict" ^
    --model "patch_outputs/model_resnet/best.pth" ^
    --output "custom_filtered.csv" ^
    --threshold 0.7
```

**重新运行 Stage 3 (可视化)**:

```bash
cd fp_filter
python visualize_filtered.py ^
    --csv "../pipeline_outputs/2026-02-10_15-30-45/stage1_wasb_detection/match1_clip1_predictions.csv" ^
    --filtered-csv "../pipeline_outputs/2026-02-10_15-30-45/stage2_fp_filtered/match1_clip1_predictions_filtered.csv" ^
    --dataset-root "../pipeline_outputs/2026-02-10_15-30-45/stage1_wasb_detection" ^
    --output-video "custom_result.mp4" ^
    --fps 30
```

**重新运行 Stage 4 (YOLO 标签生成)**:

```bash
cd fp_filter
python csv_to_yolo_txt.py ^
    --csv "../pipeline_outputs/2026-02-10_15-30-45/stage2_fp_filtered/match1_clip1_predictions_filtered.csv" ^
    --image-root "../datasets/tennis_predict" ^
    --output-dir "../pipeline_outputs/2026-02-10_15-30-45/stage4_yolo_labels/match1_clip1_predictions_filtered_yolo_labels" ^
    --box-size 15 ^
    --class-id 0
```

### 3. 性能优化建议

| 场景       | 推荐配置     | 说明                            |
| ---------- | ------------ | ------------------------------- |
| 快速预览   | `--step 5` | 每5帧检测一次，速度快但可能漏检 |
| 标准检测   | `--step 3` | 推荐配置，平衡速度和精度        |
| 高精度分析 | `--step 1` | 逐帧检测，最准确但耗时          |

---

## 与手动流程对比

### 手动流程（需要4步）

```bash
# Step 1: WASB 检测
cd src
python main.py --config-name=eval dataset=tennis_predict model=wasb ^
    detector.model_path=../pretrained_weights/wasb_tennis_best.pth.tar ^
    runner.split=test runner.vis_result=True detector.step=3

# Step 2: FP 过滤
cd ../fp_filter
python inference.py ^
    --csv "../src/outputs/main/2026-02-10_17-43-40/match1_clip1_predictions.csv" ^
    --dataset-root "../datasets/tennis_predict" ^
    --model "patch_outputs/model_resnet/best.pth" ^
    --output "patch_outputs/patches_prediction/match1_clip1_predictions_filtered.csv"

# Step 3: 可视化
python visualize_filtered.py ^
    --csv "../src/outputs/main/2026-02-10_17-43-40/match1_clip1_predictions.csv" ^
    --filtered-csv "patch_outputs/patches_prediction/match1_clip1_predictions_filtered.csv" ^
    --dataset-root "../src/outputs/main/2026-02-10_17-43-40" ^
    --output-video "patch_outputs/filtered_result.mp4" --fps 25

# Step 4: 生成 YOLO 标签
python csv_to_yolo_txt.py ^
    --csv "patch_outputs/patches_prediction/match1_clip1_predictions_filtered.csv" ^
    --image-root "../datasets/tennis_predict" ^
    --box-size 15 --class-id 0
```

### Pipeline 流程（一步完成）

```bash
python run_inference_pipeline.py
```

**优势**:

- ✅ 一行命令完成全流程
- ✅ 自动处理路径依赖关系
- ✅ 统一的输出目录管理
- ✅ 完整的日志记录
- ✅ 错误处理和状态检查

---

## 日志与调试

### 查看详细日志

脚本运行时会在控制台输出详细日志：

```
2026-02-10 15:30:45 [INFO] Pipeline 输出目录: pipeline_outputs/2026-02-10_15-30-45
2026-02-10 15:30:45 [INFO] ✓ 所有必要文件验证通过
2026-02-10 15:30:45 [INFO] ================================================================================
2026-02-10 15:30:45 [INFO] 开始执行完整推理 Pipeline
2026-02-10 15:30:45 [INFO] ================================================================================
2026-02-10 15:30:45 [INFO] 
================================================================================
2026-02-10 15:30:45 [INFO] Stage 1: WASB 球检测
2026-02-10 15:30:45 [INFO] ================================================================================
...
```

### 保存日志到文件

如需保存日志，可以使用重定向：

```bash
python run_inference_pipeline.py > pipeline.log 2>&1
```

或在 Windows PowerShell 中：

```powershell
python run_inference_pipeline.py *>&1 | Tee-Object -FilePath pipeline.log
```

---

## 性能参考

测试环境：Intel i7-10700K, NVIDIA RTX 3080, 16GB RAM

| 数据量  | Step | Stage 1 | Stage 2 | Stage 3 | Stage 4 | 总耗时    |
| ------- | ---- | ------- | ------- | ------- | ------- | --------- |
| 500 帧  | 3    | ~2 分钟 | ~30 秒  | ~1 分钟 | ~5 秒   | ~3.5 分钟 |
| 1000 帧 | 3    | ~4 分钟 | ~1 分钟 | ~2 分钟 | ~10 秒  | ~7 分钟   |
| 500 帧  | 1    | ~5 分钟 | ~30 秒  | ~1 分钟 | ~5 秒   | ~6.5 分钟 |

---

## 技术支持

如遇到问题，请检查：

1. **环境配置**: 确认 Python 版本和依赖包已正确安装
2. **数据格式**: 确认数据集目录结构符合要求
3. **模型文件**: 确认所有模型权重文件已就位
4. **日志信息**: 查看详细的错误日志定位问题

如需进一步帮助，请参考项目其他文档：

- `README.md`: 项目总体说明
- `GET_STARTED.md`: 入门指南
- `fp_filter/FP过滤二分类使用说明_UPDATED.md`: FP 过滤器详细说明
