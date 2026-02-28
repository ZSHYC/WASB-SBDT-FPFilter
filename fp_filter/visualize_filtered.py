"""
可视化过滤后的检测结果：将原始检测和过滤后的结果对比显示在图像上

使用示例：
cd fp_filter

# 可视化过滤后的结果（保存为图片）
python visualize_filtered.py ^
    --csv "../src/outputs/main/2026-02-06_16-46-34/match1_clip1_predictions.csv" ^
    --filtered-csv "patch_outputs/patches_prediction/match1_clip1_predictions_filtered.csv" ^
    --dataset-root "../src/outputs/main/2026-02-06_16-46-34" ^
    --output-dir "patch_outputs/visualizations" ^
    --sample-rate 10
    
python visualize_filtered.py --csv "../src/outputs/main/2026-02-06_17-06-39/match1_clip1_predictions.csv" --filtered-csv "patch_outputs/patches_prediction/match1_clip1_predictions_filtered.csv" --dataset-root "../src/outputs/main/2026-02-06_17-06-39" --output-dir "patch_outputs/visualizations" --sample-rate 10

# 生成对比视频
python visualize_filtered.py ^
    --csv "../src/outputs/main/2026-02-10_10-12-40/match1_clip1_predictions.csv" ^
    --filtered-csv "patch_outputs/patches_prediction/match1_clip1_predictions_filtered.csv" ^
    --dataset-root "../src/outputs/main/2026-02-10_10-12-40" ^
    --output-video "patch_outputs/filtered_result3.mp4" ^
    --fps 25
    
python visualize_filtered.py --csv "../src/outputs/main/2026-02-27_16-46-15/match1_clip1_predictions.csv" --filtered-csv "patch_outputs/patches_prediction/match1_clip1_predictions_filtered.csv" --dataset-root "../src/outputs/main/2026-02-27_16-46-15" --output-video "patch_outputs/filtered_result.mp4" --fps 25
"""

import os
import os.path as osp
import argparse
import re
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm


def _parse_match_clip_from_csv_basename(csv_basename):
    """从 CSV 文件名解析 match 和 clip"""
    m = re.match(r"^(.+?)_(.+?)_predictions", csv_basename, re.IGNORECASE)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None, None


def draw_detection(img, x, y, color, radius=8, thickness=2, label=None):
    """在图像上绘制检测点"""
    x_int, y_int = int(round(x)), int(round(y))
    cv2.circle(img, (x_int, y_int), radius, color, thickness)
    
    if label:
        # 添加文本标签
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness_text = 1
        text_size = cv2.getTextSize(label, font, font_scale, thickness_text)[0]
        
        # 文本背景
        text_x = x_int + radius + 5
        text_y = y_int
        cv2.rectangle(img, 
                     (text_x - 2, text_y - text_size[1] - 2),
                     (text_x + text_size[0] + 2, text_y + 2),
                     color, -1)
        cv2.putText(img, label, (text_x, text_y), font, font_scale, (255, 255, 255), thickness_text)
    
    return img


def visualize_images(original_csv, filtered_csv, dataset_root, output_dir, 
                     sample_rate=1, show_scores=False, match_override=None, clip_override=None):
    """
    生成可视化图片：对比原始检测和过滤后的检测
    
    Args:
        original_csv: 原始预测 CSV（或包含 fp_score 的过滤后 CSV）
        filtered_csv: 过滤后的 CSV（可选，如果不提供则从 fp_score 列判断）
        dataset_root: 图像根目录
        output_dir: 输出目录
        sample_rate: 采样率（每 N 帧保存一张图）
        show_scores: 是否显示 fp_score
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取 CSV
    df_original = pd.read_csv(original_csv) if original_csv else pd.read_csv(filtered_csv)
    df_filtered = pd.read_csv(filtered_csv) if filtered_csv else df_original
    
    # 确定图片路径
    match_name = match_override
    clip_name = clip_override
    if match_name is None or clip_name is None:
        base = osp.basename(filtered_csv or original_csv)
        p_match, p_clip = _parse_match_clip_from_csv_basename(base)
        match_name = match_name or p_match
        clip_name = clip_name or p_clip
    
    if match_name and clip_name:
        # 尝试 match_clip 格式（单层目录）
        frame_dir = osp.join(dataset_root, f"{match_name}_{clip_name}")
        if not osp.isdir(frame_dir):
            # 尝试 match/clip 格式（两层目录）
            frame_dir = osp.join(dataset_root, match_name, clip_name)
            if not osp.isdir(frame_dir):
                frame_dir = dataset_root
    else:
        frame_dir = dataset_root
    
    print(f"图像目录: {frame_dir}")
    if not osp.isdir(frame_dir):
        print(f"警告：图像目录不存在！")
    print(f"输出目录: {output_dir}")
    
    # 按文件名分组
    grouped = df_filtered.groupby("file name")
    frame_names = sorted(df_filtered["file name"].unique())
    
    saved_count = 0
    for idx, fname in enumerate(tqdm(frame_names, desc="生成可视化")):
        if idx % sample_rate != 0:
            continue
            
        img_path = osp.join(frame_dir, fname)
        if not osp.isfile(img_path):
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        vis_img = img.copy()
        
        # 获取该帧的所有检测
        frame_data = grouped.get_group(fname)
        
        for _, row in frame_data.iterrows():
            try:
                vis = int(row.get("visibility", 0))
            except:
                vis = 0
            
            x = row.get("x-coordinate")
            y = row.get("y-coordinate")
            
            if pd.isna(x) or pd.isna(y):
                continue
            
            x, y = float(x), float(y)
            
            # 检查坐标是否为有限值（排除 inf 和 -inf）
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            
            # 判断是否为有效检测
            if vis == 1:
                # 保留的检测（绿色）
                color = (0, 255, 0)  # 绿色
                label_text = "TP"
                if show_scores and "fp_score" in row and not pd.isna(row["fp_score"]):
                    score = float(row["fp_score"])
                    label_text = f"TP:{score:.2f}"
            else:
                # 被过滤的检测（红色）
                color = (0, 0, 255)  # 红色
                label_text = "FP"
                if show_scores and "fp_score" in row and not pd.isna(row["fp_score"]):
                    score = float(row["fp_score"])
                    label_text = f"FP:{score:.2f}"
            
            draw_detection(vis_img, x, y, color, label=label_text if show_scores else None)
        
        # 添加图例
        legend_y = 30
        cv2.putText(vis_img, "Green=TP(Ball), Red=FP(Filtered)", (10, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis_img, fname, (10, legend_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 保存
        output_path = osp.join(output_dir, fname)
        cv2.imwrite(output_path, vis_img)
        saved_count += 1
    
    print(f"\n完成！共保存 {saved_count} 张可视化图片到 {output_dir}")


def visualize_video(original_csv, filtered_csv, dataset_root, output_video,
                    fps=25, match_override=None, clip_override=None, show_scores=False):
    """
    生成可视化视频：对比原始检测和过滤后的检测
    """
    # 读取 CSV
    df_original = pd.read_csv(original_csv) if original_csv else pd.read_csv(filtered_csv)
    df_filtered = pd.read_csv(filtered_csv) if filtered_csv else df_original
    
    # 确定图片路径
    match_name = match_override
    clip_name = clip_override
    if match_name is None or clip_name is None:
        base = osp.basename(filtered_csv or original_csv)
        p_match, p_clip = _parse_match_clip_from_csv_basename(base)
        match_name = match_name or p_match
        clip_name = clip_name or p_clip
    
    if match_name and clip_name:
        # 尝试 match_clip 格式（单层目录）
        frame_dir = osp.join(dataset_root, f"{match_name}_{clip_name}")
        if not osp.isdir(frame_dir):
            # 尝试 match/clip 格式（两层目录）
            frame_dir = osp.join(dataset_root, match_name, clip_name)
            if not osp.isdir(frame_dir):
                frame_dir = dataset_root
    else:
        frame_dir = dataset_root
    
    print(f"图像目录: {frame_dir}")
    print(f"输出视频: {output_video}")
    
    # 按文件名分组
    grouped = df_filtered.groupby("file name")
    frame_names = sorted(df_filtered["file name"].unique())
    
    # 初始化视频写入器
    first_frame_path = osp.join(frame_dir, frame_names[0])
    first_img = cv2.imread(first_frame_path)
    if first_img is None:
        print(f"错误：无法读取第一帧 {first_frame_path}")
        return
    
    h, w = first_img.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))
    
    # 处理所有帧
    for fname in tqdm(frame_names, desc="生成视频"):
        img_path = osp.join(frame_dir, fname)
        if not osp.isfile(img_path):
            # 使用黑帧填充
            vis_img = np.zeros((h, w, 3), dtype=np.uint8)
        else:
            img = cv2.imread(img_path)
            if img is None:
                vis_img = np.zeros((h, w, 3), dtype=np.uint8)
            else:
                vis_img = img.copy()
                
                # 获取该帧的所有检测
                if fname in grouped.groups:
                    frame_data = grouped.get_group(fname)
                    
                    for _, row in frame_data.iterrows():
                        try:
                            vis = int(row.get("visibility", 0))
                        except:
                            vis = 0
                        
                        x = row.get("x-coordinate")
                        y = row.get("y-coordinate")
                        
                        if pd.isna(x) or pd.isna(y):
                            continue
                        
                        x, y = float(x), float(y)
                        
                        # 检查坐标是否为有限值（排除 inf 和 -inf）
                        if not (np.isfinite(x) and np.isfinite(y)):
                            continue
                        
                        # 判断是否为有效检测
                        if vis == 1:
                            color = (0, 255, 0)  # 绿色 - TP
                            label_text = None
                            if show_scores and "fp_score" in row and not pd.isna(row["fp_score"]):
                                score = float(row["fp_score"])
                                label_text = f"{score:.2f}"
                        else:
                            color = (0, 0, 255)  # 红色 - FP
                            label_text = None
                            if show_scores and "fp_score" in row and not pd.isna(row["fp_score"]):
                                score = float(row["fp_score"])
                                label_text = f"{score:.2f}"
                        
                        draw_detection(vis_img, x, y, color, label=label_text, radius=6, thickness=2)
        
        # 添加图例
        cv2.putText(vis_img, "Green=TP(Ball), Red=FP(Filtered)", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        out.write(vis_img)
    
    out.release()
    print(f"\n完成！视频已保存到 {output_video}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="可视化 FP Filter 过滤后的检测结果")
    parser.add_argument("--csv", default=None, help="原始检测结果 CSV（可选）")
    parser.add_argument("--filtered-csv", required=True, help="过滤后的 CSV")
    parser.add_argument("--dataset-root", required=True, help="图像根目录")
    parser.add_argument("--output-dir", default=None, help="输出图片目录（生成图片时使用）")
    parser.add_argument("--output-video", default=None, help="输出视频路径（生成视频时使用）")
    parser.add_argument("--sample-rate", type=int, default=10, help="图片模式的采样率（每N帧保存一张）")
    parser.add_argument("--fps", type=float, default=25.0, help="输出视频帧率")
    parser.add_argument("--show-scores", action="store_true", help="显示 fp_score")
    parser.add_argument("--match", default=None, help="显式指定 match 名称")
    parser.add_argument("--clip", default=None, help="显式指定 clip 名称")
    
    args = parser.parse_args()
    
    if args.output_video:
        # 生成视频
        visualize_video(
            original_csv=args.csv,
            filtered_csv=args.filtered_csv,
            dataset_root=args.dataset_root,
            output_video=args.output_video,
            fps=args.fps,
            match_override=args.match,
            clip_override=args.clip,
            show_scores=args.show_scores
        )
    elif args.output_dir:
        # 生成图片
        visualize_images(
            original_csv=args.csv,
            filtered_csv=args.filtered_csv,
            dataset_root=args.dataset_root,
            output_dir=args.output_dir,
            sample_rate=args.sample_rate,
            show_scores=args.show_scores,
            match_override=args.match,
            clip_override=args.clip
        )
    else:
        print("错误：请指定 --output-dir（生成图片）或 --output-video（生成视频）")
