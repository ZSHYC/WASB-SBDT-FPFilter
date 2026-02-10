# Pipeline 技术架构说明

## 系统概述

本 Pipeline 将网球检测系统的三个独立模块串联成一个自动化工作流，实现从原始图片到最终检测视频的端到端处理。

```
原始图片序列 → [Stage 1] → [Stage 2] → [Stage 3] → 最终检测视频
    ↓             WASB        FP过滤      可视化         ↓
数据集目录        检测         过滤        生成       结果视频
```

---

## 三阶段架构

### Stage 1: WASB 球检测 (Initial Detection)

**功能**: 使用深度学习模型对图片序列进行球体检测

**技术实现**:
- 调用 `src/main.py`（基于 Hydra 配置系统）
- 使用 WASB (Weakly-Supervised Ball Detection) 模型
- 输入: 原始图片序列
- 输出: 
  - `*_predictions.csv`: 包含每帧的检测坐标、可见性、置信度
  - 可视化图片序列: 标注了检测结果的图片

**核心配置**:
```yaml
dataset: tennis_predict           # 数据集配置
model: wasb                       # 使用 WASB 模型
detector.model_path: [权重路径]   # 预训练权重
detector.step: 3                  # 检测步长
runner.vis_result: True           # 生成可视化
```

**关键技术**:
- **Heatmap-based Detection**: 模型输出热力图，峰值位置为球的可能位置
- **Temporal Smoothing**: 利用前后帧信息平滑检测结果
- **Multi-scale Processing**: 处理不同距离和大小的球

---

### Stage 2: FP 过滤 (False Positive Filtering)

**功能**: 使用二分类模型识别并剔除误检结果

**技术实现**:
- 调用 `fp_filter/inference.py`
- 使用 ResNet-18 二分类模型
- 输入:
  - Stage 1 的 CSV 检测结果
  - 原始图片（用于提取 patch）
- 输出:
  - `*_predictions_filtered.csv`: 添加了 `fp_score` 和 `fp_pred` 列
  - 误检点的 `visibility` 被置为 0

**工作流程**:
```
对于 CSV 中每个 visibility=1 的检测点:
  1. 从原始图片中提取以 (x,y) 为中心的 128×128 patch
  2. 将 patch 输入二分类模型
  3. 获得 "假阳性概率" (fp_score)
  4. 如果 fp_score > threshold:
       - 标记为误检 (fp_pred=1)
       - 将 visibility 置为 0
  5. 否则保留该检测点
```

**模型架构**:
- **Backbone**: ResNet-18 (ImageNet 预训练)
- **输入**: 128×128 RGB patch
- **输出**: 2-class softmax (真球 vs 误检)

**Patch 提取**:
```python
# 伪代码
center_x, center_y = detection_point
half_size = patch_size // 2
patch = image[
    center_y - half_size : center_y + half_size,
    center_x - half_size : center_x + half_size
]
```

---

### Stage 3: 结果可视化 (Visualization)

**功能**: 生成对比视频，直观展示原始检测和过滤后的结果

**技术实现**:
- 调用 `fp_filter/visualize_filtered.py`
- 输入:
  - Stage 1 的原始 CSV
  - Stage 2 的过滤后 CSV
  - Stage 1 生成的可视化图片
- 输出:
  - MP4 视频文件

**可视化方案**:
```
对于每一帧:
  1. 读取原始检测 (从 original_csv)
  2. 读取过滤后检测 (从 filtered_csv)
  3. 在图片上绘制:
     - 红色圆圈: 所有原始检测点
     - 绿色圆圈: 过滤后保留的检测点
  4. 将标注后的图片写入视频流
```

**视频编码**:
- 使用 OpenCV VideoWriter
- 编码格式: MP4 (H.264)
- 可自定义帧率 (默认 25 fps)

---

## 数据流图

```
[原始数据集]
    │
    ├─── datasets/tennis_predict/
    │        └─── match1/
    │                └─── clip1/
    │                      ├─── 0001.jpg
    │                      ├─── 0002.jpg
    │                      └─── ...
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: WASB Detection                                     │
│ [src/main.py + WASB Model]                                  │
└─────────────────────────────────────────────────────────────┘
    │
    ├─── match1_clip1_predictions.csv
    │      ├─── file_name, x, y, visibility, score
    │      ├─── 0001.jpg, 520.3, 315.7, 1, 0.92
    │      ├─── 0002.jpg, 525.1, 318.2, 1, 0.88
    │      └─── ...
    │
    ├─── match1_clip1/ (可视化图片)
    │      ├─── 0001.jpg
    │      ├─── 0002.jpg
    │      └─── ...
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 2: FP Filtering                                       │
│ [fp_filter/inference.py + ResNet-18 Classifier]            │
└─────────────────────────────────────────────────────────────┘
    │
    ├─── match1_clip1_predictions_filtered.csv
    │      ├─── file_name, x, y, visibility, score, fp_score, fp_pred
    │      ├─── 0001.jpg, 520.3, 315.7, 1, 0.92, 0.15, 0  ← 保留
    │      ├─── 0002.jpg, 525.1, 318.2, 0, 0.88, 0.85, 1  ← 过滤掉
    │      └─── ...
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 3: Visualization                                      │
│ [fp_filter/visualize_filtered.py + OpenCV]                 │
└─────────────────────────────────────────────────────────────┘
    │
    └─── match1_clip1_final_result.mp4
          └─── 对比视频（红圈=原始，绿圈=过滤后）
```

---

## 关键技术点

### 1. 路径管理

Pipeline 脚本使用 `pathlib.Path` 进行跨平台路径处理：

```python
# 项目根目录
root_dir = Path(__file__).parent.absolute()

# 相对路径转绝对路径
dataset_root = root_dir / args.dataset_root
wasb_weight = root_dir / args.wasb_weight

# 输出目录自动添加时间戳
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = root_dir / args.output_base / timestamp
```

### 2. 子进程调用

使用 `subprocess` 调用各阶段脚本，确保环境隔离：

```python
result = subprocess.run(
    cmd,                    # 命令列表
    cwd=str(work_dir),      # 工作目录
    check=True,             # 出错时抛出异常
    capture_output=True,    # 捕获输出
    text=True,              # 文本模式
    encoding='utf-8'        # UTF-8 编码
)
```

### 3. Hydra 配置覆盖

Stage 1 使用 Hydra 配置系统，通过命令行参数覆盖：

```python
cmd = [
    "python", "main.py",
    "--config-name=eval",                    # 基础配置
    "dataset=tennis_predict",                # 覆盖数据集
    "detector.model_path=path/to/model",     # 覆盖模型路径
    f"hydra.run.dir={output_dir}",          # 指定输出目录
]
```

### 4. CSV 文件匹配

自动查找和匹配不同阶段的 CSV 文件：

```python
# Stage 1 输出
csv_files = list(stage1_output.glob("*_predictions.csv"))

# Stage 2: 对应的过滤后文件
for csv_file in csv_files:
    filtered_name = csv_file.name.replace("_predictions.csv", 
                                          "_predictions_filtered.csv")
    output_csv = stage2_output / filtered_name
```

---

## 性能优化

### 1. 批处理优化

WASB 模型使用批处理提高 GPU 利用率：

```python
# dataloader 配置
batch_size: 8           # 根据 GPU 内存调整
num_workers: 4          # 数据加载并行度
```

### 2. 步长调整

通过 `--step` 参数控制检测密度：

| Step | 含义 | 性能 | 精度 |
|------|------|------|------|
| 1 | 逐帧检测 | 慢 | 最高 |
| 3 | 每3帧检测 | 中 | 高 |
| 5 | 每5帧检测 | 快 | 中 |

### 3. GPU 加速

所有深度学习模型自动使用 GPU（如果可用）：

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

---

## 错误处理机制

### 1. 文件验证

Pipeline 启动前验证所有必需文件：

```python
def _validate_prerequisites(self):
    if not self.dataset_root.exists():
        raise FileNotFoundError(f"数据集目录不存在: {self.dataset_root}")
    # ... 其他验证
```

### 2. 阶段依赖检查

每个阶段检查前序阶段的输出：

```python
csv_files = list(self.stage1_output.glob("*_predictions.csv"))
if not csv_files:
    log.error("Stage 1 未生成任何 CSV 文件")
    return False
```

### 3. 异常捕获

使用 try-except 捕获子进程错误：

```python
try:
    result = subprocess.run(cmd, check=True, ...)
except subprocess.CalledProcessError as e:
    log.error(f"Stage 失败: {e}")
    log.error(f"错误输出: {e.stderr}")
    return False
```

---

## 扩展性设计

### 1. 模块化架构

每个 Stage 都是独立的方法，便于维护和扩展：

```python
class InferencePipeline:
    def _run_stage1_wasb_detection(self): ...
    def _run_stage2_fp_filtering(self): ...
    def _run_stage3_visualization(self): ...
```

### 2. 参数化配置

所有关键参数都可通过命令行传入：

```python
parser.add_argument('--dataset-root', ...)
parser.add_argument('--wasb-weight', ...)
parser.add_argument('--threshold', ...)
```

### 3. 多数据集支持

自动处理多个 match/clip：

```python
# 自动查找所有 CSV 文件
csv_files = list(stage1_output.glob("*_predictions.csv"))

# 逐个处理
for csv_file in csv_files:
    # 处理单个文件
    ...
```

---

## 与原有脚本的兼容性

Pipeline 脚本**不修改**任何原有脚本，完全通过参数传递实现功能：

| 原脚本 | 调用方式 | 修改 |
|--------|---------|------|
| `src/main.py` | subprocess + Hydra 参数 | ❌ 无需修改 |
| `fp_filter/inference.py` | subprocess + 命令行参数 | ❌ 无需修改 |
| `fp_filter/visualize_filtered.py` | subprocess + 命令行参数 | ❌ 无需修改 |

---

## 未来改进方向

### 1. 并行处理

对多个 match/clip 进行并行处理：

```python
from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(process_clip, csv) for csv in csv_files]
```

### 2. 进度跟踪

添加更精细的进度条：

```python
from tqdm import tqdm

for csv_file in tqdm(csv_files, desc="Processing clips"):
    # 处理
    ...
```

### 3. 结果分析

自动生成统计报告：

```python
def _generate_report(self):
    # 统计原始检测数、过滤后数量、过滤率等
    report = {
        "total_detections": ...,
        "filtered_out": ...,
        "filter_rate": ...,
    }
    # 保存为 JSON 或 PDF
```

### 4. Web 界面

开发基于 Web 的界面，提供可视化配置和结果查看：

```python
# 使用 Flask/FastAPI
@app.post("/run_pipeline")
def run_pipeline(config: PipelineConfig):
    pipeline = InferencePipeline(config)
    return pipeline.run()
```

---

## 总结

本 Pipeline 系统的核心优势：

1. ✅ **自动化**: 一行命令完成三个步骤
2. ✅ **可追溯**: 带时间戳的目录管理，所有中间结果都保留
3. ✅ **可配置**: 丰富的参数选项，适应不同场景
4. ✅ **兼容性**: 不修改原有脚本，完全向后兼容
5. ✅ **健壮性**: 完善的错误处理和日志记录
6. ✅ **可扩展**: 模块化设计，易于添加新功能

通过将三个独立脚本串联，大幅提升了系统的易用性和工程化水平。
