# 完整推理 Pipeline 文档导航

欢迎使用网球检测完整推理 Pipeline！本目录包含了从原始图片到最终检测视频的一站式自动化解决方案。

---

## 📚 文档索引

根据你的需求选择对应的文档：

### 🚀 快速开始

- **[PIPELINE_快速参考.md](PIPELINE_快速参考.md)** 
  - 一页纸速查表
  - 最常用的命令
  - 参数速查表

### 📖 详细文档

- **[PIPELINE_使用说明.md](PIPELINE_使用说明.md)**
  - 完整的使用教程
  - 所有参数详解
  - 常见问题排查
  - 使用示例

### 🏗️ 技术文档

- **[PIPELINE_技术架构.md](PIPELINE_技术架构.md)**
  - 系统架构详解
  - 三阶段工作流程
  - 数据流图
  - 扩展开发指南

### 💡 配置示例

- **[pipeline_config_examples.txt](pipeline_config_examples.txt)**
  - 不同场景的配置模板
  - 参数调优建议
  - 复制即用的命令

---

## 🎯 快速入门（30秒上手）

### 1. 准备数据

将图片放在正确位置：
```
datasets/tennis_predict/match1/clip1/*.jpg
```

### 2. 一行命令运行

```bash
# Linux/Mac
python run_inference_pipeline.py

# Windows (双击运行)
run_pipeline.bat
```

### 3. 查看结果

```
pipeline_outputs/
└── [时间戳]/
    └── stage3_visualizations/
        └── *.mp4  ← 这里就是最终视频！
```

---

## 🔧 核心文件说明

| 文件 | 用途 | 适用人群 |
|------|------|---------|
| `run_inference_pipeline.py` | Pipeline 主脚本 | 所有用户 |
| `run_pipeline.bat` | Windows 一键启动脚本 | Windows 用户 |
| `PIPELINE_快速参考.md` | 快速参考卡片 | 熟悉用户 |
| `PIPELINE_使用说明.md` | 完整使用手册 | 新手用户 |
| `PIPELINE_技术架构.md` | 技术架构文档 | 开发者 |
| `pipeline_config_examples.txt` | 配置示例集 | 所有用户 |

---

## 🎬 工作流程一览

```
原始图片        WASB检测         FP过滤          可视化         最终视频
  📁    →    🔍 Stage 1   →   🎯 Stage 2   →   📹 Stage 3   →    🎬
dataset      predictions      filtered         video          result.mp4
```

---

## 💬 常见问题

### Q1: 我该从哪里开始？

**A**: 按以下顺序阅读文档：
1. [PIPELINE_快速参考.md](PIPELINE_快速参考.md) - 快速了解用法
2. [PIPELINE_使用说明.md](PIPELINE_使用说明.md) - 详细学习

### Q2: 运行出错怎么办？

**A**: 
1. 查看 [PIPELINE_使用说明.md](PIPELINE_使用说明.md) 的"常见问题排查"章节
2. 检查日志输出中的错误信息
3. 确认前置条件是否满足

### Q3: 如何调整参数？

**A**: 
- 参考 [pipeline_config_examples.txt](pipeline_config_examples.txt) 中的场景化配置
- 或查看 [PIPELINE_使用说明.md](PIPELINE_使用说明.md) 的"参数说明"章节

### Q4: 如何理解系统原理？

**A**: 
- 阅读 [PIPELINE_技术架构.md](PIPELINE_技术架构.md) 了解完整的技术细节

---

## 🆘 获取帮助

### 命令行帮助

```bash
python run_inference_pipeline.py --help
```

### 在线文档

- **FP 过滤器详细说明**: [fp_filter/FP过滤二分类使用说明_UPDATED.md](fp_filter/FP过滤二分类使用说明_UPDATED.md)
- **项目整体说明**: [README.md](README.md)
- **入门指南**: [GET_STARTED.md](GET_STARTED.md)

---

## 📊 性能参考

| 数据量 | 配置 | 预计耗时 |
|-------|------|---------|
| 500帧 | 标准模式 (step=3) | ~3.5分钟 |
| 1000帧 | 标准模式 (step=3) | ~7分钟 |
| 500帧 | 高精度模式 (step=1) | ~6.5分钟 |

*测试环境: Intel i7-10700K, NVIDIA RTX 3080, 16GB RAM*

---

## 🎉 特性亮点

✅ **一键运行**: 一行命令完成全流程  
✅ **自动化管理**: 自动处理路径和依赖关系  
✅ **批量处理**: 自动处理多个 match/clip  
✅ **完整日志**: 详细的执行日志和错误信息  
✅ **可追溯性**: 带时间戳的输出目录  
✅ **灵活配置**: 丰富的参数选项  
✅ **跨平台**: 支持 Windows/Linux/Mac  

---

## 📝 反馈与改进

如有问题或建议，欢迎反馈！

**项目版本**: v1.0  
**最后更新**: 2026-02-10  
**维护者**: GitHub Copilot  

---

## 🔗 相关链接

- **WASB 项目**: [nttcom/WASB-SBDT](https://github.com/nttcom/WASB-SBDT)
- **模型权重下载**: 参考 [MODEL_ZOO.md](MODEL_ZOO.md)

---

**祝使用愉快！Have fun with automated ball detection! 🎾**
