# WASB 项目说明：训练流程 + 不改代码下的预测精度提升

本文档说明：(1) 项目训练过程；(2) 仅使用现有 WASB 权重、**不修改任何代码**的前提下，可通过**命令行/配置文件**调整哪些参数来提高预测精度。

---

## 一、项目与 WASB 模型简要说明

- **项目**：WASB-SBDT-FPFilter（Widely Applicable Strong Baseline for Sports Ball Detection and Tracking + FP Filter），用于多类运动的球检测与跟踪。
- **你只用 WASB 模型**：WASB 基于 HRNet，输入为连续多帧图像，输出为热图，再经后处理得到球心坐标与可见性。
- **入口**：`src/main.py` 默认用 Hydra 加载 `config_name=eval`，即**评估/推理**流程（不训练）。

---

## 二、训练过程是怎样的（仅作了解，你不改训练）

- 仓库中 **GET_STARTED.md** 里 “Training” 写的是 **TBA**，即官方未在文档中给出完整训练步骤。
- 从代码结构可知：
  - **训练**由 `runners/train_and_test.py` 里的 `Trainer` 完成，会调用 `build_dataloader`、`build_model`、`build_loss_criteria`、`build_optimizer_and_scheduler`，按 `runner.max_epochs` 等配置跑多轮训练，并在指定 epoch 做验证/推理视频。
  - **评估/推理**由 `runners/eval.py` 里的 `VideosInferenceRunner` 完成：按 clip 加载数据 → 用 detector 对每批图像跑前向 → 后处理得到检测框/点 → 再用 **tracker** 做跨帧关联，得到每帧的球位置与可见性，若有 GT 则用 **Evaluator** 算 Precision、Recall、F1、Accuracy、RMSE 等。

你当前是**只做评估/推理、使用已有权重**，因此下面只围绕“**不改代码，只调配置/命令行**”的提精度措施展开。

---

## 三、推理/评估流程（你实际在用的）

1. **数据**：按 `dataset` 配置（如 `tennis` 或 `tennis_predict`）加载 test 的 clips，每个样本为连续 `frames_in=3` 帧。
2. **模型**：WASB（HRNet），输入尺寸由 `model.inp_height/inp_width`（如 288×512）决定，输出多通道热图。
3. **检测器**：对热图做 sigmoid → 根据 **score_threshold** 二值化 → 用 **blob_det_method**（concomp 或 nms）找连通分量/峰值 → 得到 (x,y) 和 score，并可用 **use_hm_weight** 做质心加权。
4. **跟踪器**：`online` tracker 按 **max_disp** 做帧间关联，得到每帧的最终 (x, y, visi, score)。
5. **评估**：若提供了 GT，则用 **runner.eval.dist_threshold** 判断“预测是否算对”，并汇总 Prec、Recall、F1、Accuracy、RMSE 等。

以下所有“可调项”都是**通过 Hydra 命令行覆盖或改 YAML** 即可，无需改 Python 代码。

---

## 四、不改代码可调参数一览（提高预测精度可做的措施）

### 4.1 检测器相关（对精度影响最大）

这些参数在 **detector** 配置下，主要来自 `configs/detector/tracknetv2.yaml`（WASB 评估时用的就是 tracknetv2 这套 detector 配置）。

| 参数                                             | 默认值  | 含义                                                                   | 调参建议                                                                                                                                                                                                                      |
| ------------------------------------------------ | ------- | ---------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **detector.model_path**                    | 无      | 权重文件路径                                                           | 必须指定，如 `../pretrained_weights/wasb_tennis_best.pth.tar`。                                                                                                                                                             |
| **detector.step**                          | 3       | 推理步长（每多少帧做一次模型推理）                                     | 论文中 Step=1 比 Step=3 更准、更慢。**想提高精度可设为 1**：`detector.step=1`。                                                                                                                                             |
| **detector.postprocessor.score_threshold** | 0.5     | 热图二值化阈值，高于此值才认为有球                                     | **漏检多**时可适当**降低**（如 0.3～0.45），会多检出球、提高召回、可能增加误检；**误检多**时可**提高**（如 0.55～0.65），会少检出、提高精确率、可能漏检。需根据你的数据在 0.3～0.6 之间做小步长尝试。 |
| **detector.postprocessor.use_hm_weight**   | True    | 是否用热图值加权求质心                                                 | 保持**True** 一般比 False 定位更准。                                                                                                                                                                                    |
| **detector.postprocessor.blob_det_method** | concomp | 从二值热图提取位置的方法：`concomp`=连通分量，`nms`=局部极大值+NMS | 若热图经常出现多峰或碎片，可尝试**nms**（会用到 dataloader 的 `heatmap.sigmas`）。多数情况 concomp 即可。                                                                                                             |
| **detector.postprocessor.scales**          | [0]     | 使用的输出尺度                                                         | 单尺度一般保持 [0]。                                                                                                                                                                                                          |

**命令行示例（只演示覆盖，不改变代码）：**

```powershell
# 步长改为 1，提高精度（更慢）
python main.py --config-name=eval dataset=tennis_predict model=wasb detector.model_path=../pretrained_weights/wasb_tennis_best.pth.tar detector.step=1

# 同时调低阈值，提高召回
python main.py --config-name=eval dataset=tennis_predict model=wasb detector.model_path=../pretrained_weights/wasb_tennis_best.pth.tar detector.step=1 detector.postprocessor.score_threshold=0.4
```

---

### 4.2 跟踪器相关（影响轨迹连续性、误跟/丢跟）

| 参数                       | 默认值 | 含义                                     | 调参建议                                                                                                                                                                                           |
| -------------------------- | ------ | ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **tracker.max_disp** | 300    | 相邻两帧之间，允许关联的最大位移（像素） | 球速很快、经常“跨帧跳得很远”时，可**适当增大**（如 350～400），减少因位移超限而断轨迹；若经常把别的物体误跟成球，可**适当减小**（如 200～250），让只有离上一帧位置近的检测才被连上。 |

**命令行示例：**

```powershell
python main.py --config-name=eval dataset=tennis_predict model=wasb detector.model_path=../pretrained_weights/wasb_tennis_best.pth.tar tracker.max_disp=350
```

---

### 4.3 评估相关（只影响“指标怎么算”，不改变预测结果）

| 参数                                  | 默认值 | 含义                                                                                                       | 调参建议                                                                                                  |
| ------------------------------------- | ------ | ---------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| **runner.eval.dist_threshold**  | 4      | 判定“预测正确”的距离阈值（像素）：预测与 GT 距离小于此值算 TP                                            | **不改变模型输出**，只影响 Prec/Recall/F1 等数值。想更严格可调小（如 3），更宽松可调大（如 5～6）。 |
| **runner.eval.score_threshold** | 0.5    | 在 runner 配置里存在，当前 Evaluator 实现中未使用；实际起作用的是 detector.postprocessor.score_threshold。 | 若后续代码用到，再考虑与 postprocessor 的 score_threshold 一致。                                          |

---

### 4.4 数据与运行方式

| 参数                                            | 含义                                 | 建议                                                                                                                                              |
| ----------------------------------------------- | ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| **dataset**                               | 数据集名，如 tennis / tennis_predict | 预测用哪个数据集、哪个目录下的 clips，由这里和对应 dataset 的 YAML 决定（如 `dataset/tennis_predict.yaml` 的 `root_dir`、`test.matches`）。 |
| **runner.split**                          | train / test                         | 评估 test 集用 `runner.split=test`。                                                                                                            |
| **runner.vis_result / vis_hm / vis_traj** | 是否保存可视化结果、热图、轨迹       | 调试时可设为 True，便于看漏检/误检发生在哪一帧。                                                                                                  |
| **runner.gpus**                           | 如 [0]                               | 按你机器上的 GPU 编号设置。                                                                                                                       |

---

### 4.5 模型输入尺寸（wasb.yaml）

- **model.inp_height / model.inp_width**：如 288、512。
- 推理时图像会被 **resize** 到该尺寸再送进网络；**不改代码**的情况下，这些值由 `configs/model/wasb.yaml` 决定，一般**不要改**（和预训练权重一致）。若你将来通过改配置改成别的尺寸，需要确认与预训练一致，否则可能掉点。

---

### 4.6 Dataloader / Heatmap（对推理的间接影响）

- **dataloader.heatmap.sigmas**：在 **blob_det_method=nms** 时，后处理用该 sigma 做 NMS 邻域；**concomp** 时不影响。
- 推理阶段**不生成 GT 热图**，只做前向与后处理，因此 mags、min_value 等对推理无影响。

---

## 五、推荐组合：在不改代码的前提下尽量提高预测精度

1. **必做**

   - 使用正确权重：`detector.model_path=../pretrained_weights/wasb_tennis_best.pth.tar`（或你实际路径）。
   - 使用对应数据集：如 `dataset=tennis_predict` 或 `dataset=tennis`，并保证 `root_dir`、`test.matches` 指向你的数据。
2. **优先尝试（对精度影响大）**

   - **detector.step=1**：每帧都推理，精度通常优于 step=3，代价是速度变慢。
   - **detector.postprocessor.score_threshold**：根据你的结果微调；漏检多就略降（如 0.4），误检多就略升（如 0.55）。
   - 保持 **detector.postprocessor.use_hm_weight=True**。
3. **按现象微调**

   - 轨迹经常断：适当 **增大 tracker.max_disp**。
   - 误跟、轨迹乱：适当 **减小 tracker.max_disp**。
   - 热图多峰/碎点：可试 **detector.postprocessor.blob_det_method=nms**（并确认 dataloader.heatmap.sigmas 合理）。
4. **评估指标**

   - 若你更在意“多严才算对”，可调 **runner.eval.dist_threshold**（仅改指标，不改预测）。
5. **数据与可视化**

   - 保证 **root_dir**、**test.matches**、**csv_filename** 等与你的目录和标注一致。
   - 用 **runner.vis_result=True**（以及 vis_hm/vis_traj）观察问题帧，再反过来调 score_threshold / step / max_disp。

---

## 六、一条完整的“高精度”推理示例命令（仅改参数、不改代码）

在项目根目录下（或从 `src` 运行时相应改路径），例如：

```powershell
cd src
python main.py --config-name=eval ^
  dataset=tennis_predict ^
  model=wasb ^
  detector.model_path=../pretrained_weights/wasb_tennis_best.pth.tar ^
  detector.step=1 ^
  detector.postprocessor.score_threshold=0.45 ^
  runner.split=test ^
  runner.vis_result=True ^
  runner.eval.dist_threshold=4 ^
  tracker.max_disp=300
```

可根据你的漏检/误检情况，只调整 **score_threshold**、**max_disp** 等，无需动任何 .py 文件。

---

## 七、小结

- **训练**：项目具备训练逻辑（Trainer + train_and_test），但官方文档未写具体训练步骤；你当前只使用现有 WASB 权重，无需改训练过程。
- **预测精度**：在不改代码的前提下，可通过 Hydra 覆盖：**detector.step=1**、**detector.postprocessor.score_threshold**、**detector.postprocessor.use_hm_weight**、**detector.postprocessor.blob_det_method**、**tracker.max_disp**，以及 **runner.eval.dist_threshold**（仅影响指标）。
- 建议优先做：**step=1** + **score_threshold** 小范围网格搜索（如 0.35～0.55）+ 必要时调 **max_disp**，再配合可视化看问题帧，即可在现有模型权重下尽量提高预测精度。
