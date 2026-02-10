# Pipeline 快速参考卡

## 一行命令完成全流程

```bash
python run_inference_pipeline.py
```

## 常用配置

### 快速模式（推荐日常使用）
```bash
python run_inference_pipeline.py --step 3 --threshold 0.5
```

### 高精度模式（重要比赛分析）
```bash
python run_inference_pipeline.py --step 1 --threshold 0.3
```

### 严格过滤模式（减少误检）
```bash
python run_inference_pipeline.py --step 3 --threshold 0.7
```

## 输出位置

```
pipeline_outputs/
└── [时间戳]/
    ├── stage1_wasb_detection/      ← 原始检测结果
    ├── stage2_fp_filtered/         ← 过滤后结果
    └── stage3_visualizations/      ← 最终视频 🎬
```

## 参数速查

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--step` | 3 | 检测步长 (1=最精确, 5=最快速) |
| `--threshold` | 0.5 | 过滤阈值 (越高越严格) |
| `--fps` | 25 | 输出视频帧率 |

## 前置条件检查清单

- [ ] 数据已放在 `datasets/tennis_predict/match1/clip1/`
- [ ] WASB 模型已下载到 `pretrained_weights/wasb_tennis_best.pth.tar`
- [ ] FP 模型已训练好在 `fp_filter/patch_outputs/model_resnet/best.pth`
- [ ] Python 环境已安装所有依赖包

## 查看详细说明

```bash
python run_inference_pipeline.py --help
```

或阅读 [PIPELINE_使用说明.md](PIPELINE_使用说明.md)
