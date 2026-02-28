"""
推理脚本：使用训练好的二分类模型（FP Filter）对检测结果 CSV 进行过滤。
该脚本会读取检测结果，提取对应的 patch，输入二分类模型，
若模型判定为非球（FP），则将其 visibility 置为 0 或降低置信度。

使用示例：
Step 1: 确保已训练好模型（例如 patch_outputs/fp_filter/best.pth）
Step 2: 运行以下命令（请根据实际路径修改参数）

默认文件夹在fp_filter下执行：cd fp_filter(别忘了！)
python inference.py ^
    --csv "../src/outputs/main/2026-02-10_10-12-40/match1_clip1_predictions.csv" ^
    --dataset-root "../datasets/tennis_predict" ^
    --model "patch_outputs/model_resnet/best.pth" ^
    --output "patch_outputs/patches_prediction/match1_clip1_predictions_filtered.csv" ^
    --threshold 0.5
    
python inference.py --csv "../src/outputs/main/2026-02-27_16-46-15/match1_clip1_predictions.csv" --dataset-root "../datasets/tennis_predict" --model "patch_outputs/model_resnet/best.pth" --output "patch_outputs/patches_prediction/match1_clip1_predictions_filtered.csv" --threshold 0.5
"""

import os
import os.path as osp
import argparse
import re
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

# 尝试导入同目录下的 model.py
try:
    from model import build_model
except ImportError:
    # 如果运行路径导致无法直接 import，尝试相对导入或追加路径
    import sys
    sys.path.append(osp.dirname(osp.abspath(__file__)))
    from model import build_model

# 计算机视觉中常用的小目标 patch 尺寸
DEFAULT_PATCH_SIZE = 128
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_PATCH_SIZE = 128

def _parse_match_clip_from_csv_basename(csv_basename):
    """从 CSV 文件名解析 match 和 clip"""
    m = re.match(r"^(.+?)_(.+?)_predictions\.csv$", csv_basename, re.IGNORECASE)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None, None


def get_transform(patch_size = DEFAULT_PATCH_SIZE):
    """推理时的预处理：Resize + ToTensor + 归一化（必须与训练时验证集的 transform 一致）"""
    return T.Compose([
        T.Resize((patch_size, patch_size)),  # 关键：与训练时验证集保持一致
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


@torch.no_grad()
def run_inference(
    csv_path,
    dataset_root,
    model_path,
    output_path,
    patch_size=DEFAULT_PATCH_SIZE,
    threshold=0.5,
    match_override=None,
    clip_override=None,
    device_name="cuda"
):
    print(f"正在处理 CSV: {csv_path}")
    print(f"使用模型: {model_path}")
    
    # 1. 加载模型
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    model = build_model(num_classes=2).to(device)
    
    if not osp.isfile(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
    checkpoint = torch.load(model_path, map_location=device)
    # 支持加载完整模型或仅 state_dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint) # 假设直接是 state_dict
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    transform = get_transform(patch_size)

    # 2. 读取 CSV
    df = pd.read_csv(csv_path)
    
    # 3. 确定图片路径
    match_name = match_override
    clip_name = clip_override
    if match_name is None or clip_name is None:
        base = osp.basename(csv_path)
        p_match, p_clip = _parse_match_clip_from_csv_basename(base)
        match_name = match_name or p_match
        clip_name = clip_name or p_clip
    
    if not match_name or not clip_name:
        print("[警告] 无法从文件名解析 match/clip，且未指定 override。将尝试直接在 dataset_root 下查找图片。")
        frame_dir = dataset_root
    else:
        frame_dir = osp.join(dataset_root, match_name, clip_name)
    
    print(f"图片根目录: {frame_dir}")

    # 4. 遍历检测结果并过滤
    # 新增列：fp_score (模型预测为球的概率), original_vis
    if "fp_score" not in df.columns:
        df["fp_score"] = -1.0
    
    # 为了效率，可以按图片分组处理，或者逐个处理（简单起见先逐个处理）
    valid_count = 0
    filtered_count = 0
    
    half = patch_size // 2

    for idx, row in df.iterrows():
        # 仅处理 visibility=1 的行
        try:
            vis = int(row.get("visibility", 0))
        except:
            vis = 0
            
        if vis != 1:
            continue
            
        x_center = row.get("x-coordinate")
        y_center = row.get("y-coordinate")
        fname = row.get("file name")
        
        if pd.isna(x_center) or pd.isna(y_center) or pd.isna(fname):
            continue
            
        x_center, y_center = float(x_center), float(y_center)
        img_path = osp.join(frame_dir, fname)
        
        if not osp.isfile(img_path):
            # 尝试不带 match/clip 目录
            if osp.isfile(osp.join(dataset_root, fname)):
                img_path = osp.join(dataset_root, fname)
            else:
                # 找不到图片，跳过
                continue
        
        # 读取图片并提取 patch
        # 优化：如果在同一张图上有多个检测，重复读取会慢。但对于 CSV 顺序通常是按帧排序的，可以用简易缓存。
        # 这里为了代码简洁，暂不加缓存，直接读取
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        h, w = img.shape[:2]
        x_int, y_int = int(round(x_center)), int(round(y_center))
        
        x1 = x_int - half
        y1 = y_int - half
        x2 = x1 + patch_size
        y2 = y1 + patch_size
        
        # 边界填充处理（使用 BORDER_REPLICATE 与训练数据生成保持一致）
        pad_l = max(0, -x1)
        pad_t = max(0, -y1)
        pad_r = max(0, x2 - w)
        pad_b = max(0, y2 - h)
        
        if any([pad_l, pad_t, pad_r, pad_b]):
            # ✅ 改为 BORDER_REPLICATE，与 extract_patches.py 保持一致
            patch = cv2.copyMakeBorder(img, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE)
            # 坐标平移
            x1 += pad_l
            y1 += pad_t
            x2 += pad_l
            y2 += pad_t
            patch = patch[y1:y2, x1:x2]
        else:
            patch = img[y1:y2, x1:x2]
            
        # 注意：不在这里 resize，而是在 transform 中统一处理
        # if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
        #     patch = cv2.resize(patch, (patch_size, patch_size))
            
        # 转换并推理
        patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        patch_pil = Image.fromarray(patch_rgb)
        input_tensor = transform(patch_pil).unsqueeze(0).to(device)
        
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        prob_ball = probs[0, 1].item() # 类别1为球
        
        df.at[idx, "fp_score"] = prob_ball
        
        # 这里可以直接修改 visibility，也可以只标记
        if prob_ball < threshold:
            df.at[idx, "visibility"] = 0 # 标记为不可见（过滤掉）
            filtered_count += 1
        
        valid_count += 1
        if valid_count % 100 == 0:
            print(f"已处理 {valid_count} 个检测点...", end="\r")

    print(f"\n处理完成。共检查 {valid_count} 个检测点，过滤掉 {filtered_count} 个 FP (阈值={threshold})。")
    print(f"结果保存至: {output_path}")
    
    os.makedirs(osp.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 FP Filter 模型过滤检测结果")
    parser.add_argument("--csv", required=True, help="检测结果 CSV 路径")
    parser.add_argument("--dataset-root", required=True, help="原始数据集根目录")
    parser.add_argument("--model", required=True, help="训练好的二分类模型路径 (.pth)")
    parser.add_argument("--output", required=True, help="输出过滤后的 CSV 路径")
    parser.add_argument("--patch-size", type=int, default=128, help="patch 尺寸，需与训练时一致（默认 128）")
    parser.add_argument("--threshold", type=float, default=0.5, help="认定为球的概率阈值，低于此值视为 FP")
    parser.add_argument("--match", default=None, help="显式指定 match 名称")
    parser.add_argument("--clip", default=None, help="显式指定 clip 名称")
    
    args = parser.parse_args()
    
    run_inference(
        csv_path=args.csv,
        dataset_root=args.dataset_root,
        model_path=args.model,
        output_path=args.output,
        patch_size=args.patch_size,
        threshold=args.threshold,
        match_override=args.match,
        clip_override=args.clip
    )
