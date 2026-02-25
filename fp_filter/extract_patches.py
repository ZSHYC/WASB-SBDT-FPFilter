"""
第一步：根据检测结果 CSV（visibility=1）从原始数据集中以 (x,y) 为中心截取 patch，
并生成 manifest 供后续标注与二分类训练使用。

使用示例（在 fp_filter 下执行）： cd fp_filter(别忘了！) 千万别忘了改文件名！不然就覆盖了！！！
 python extract_patches.py "../src/outputs/main/2026-02-25_10-57-05/match1_clip1_predictions.csv" --dataset-root ../datasets/tennis_predict --output-dir patch_outputs/patches_train14_1 --patch-size 128
"""
import os
import os.path as osp
import argparse
import re
import numpy as np
import pandas as pd
import cv2


# 计算机视觉中常用的小目标 patch 尺寸（便于下采样且保留上下文）
DEFAULT_PATCH_SIZE = 128


def _parse_match_clip_from_csv_basename(csv_basename):
    """从 CSV 文件名解析 match 和 clip，例如 match1_clip1_predictions.csv -> ('match1', 'clip1')"""
    # 支持 match1_clip1_predictions.csv 或 match_1_clip_1_predictions.csv 等
    m = re.match(r"^(.+?)_(.+?)_predictions\.csv$", csv_basename, re.IGNORECASE)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None, None


def _is_valid_row(row):
    """判断该行是否为 visibility=1 且坐标有效（非 -inf/nan）"""
    try:
        vis = int(row.get("visibility", 0))
    except (ValueError, TypeError):
        return False
    if vis != 1:
        return False
    x, y = row.get("x-coordinate"), row.get("y-coordinate")
    if pd.isna(x) or pd.isna(y):
        return False
    try:
        xf, yf = float(x), float(y)
    except (TypeError, ValueError):
        return False
    if np.isinf(xf) or np.isinf(yf):
        return False
    return True


def extract_patches(
    predictions_csv,
    dataset_root,
    output_dir,
    patch_size=DEFAULT_PATCH_SIZE,
    match_override=None,
    clip_override=None,
):
    """
    从 predictions_csv 中读取 visibility=1 的检测，在 dataset_root 下按 match/clip/filename
    定位原图，以 (x,y) 为中心截取 patch_size x patch_size 的 patch，保存到 output_dir，
    并生成 manifest CSV（含 patch_id, source_file, x, y, match, clip, patch_path，预留 label 列）。
    若 match/clip 未通过 override 指定，则从 CSV 文件名解析（如 match1_clip1_predictions.csv）。
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(predictions_csv)

    match_name = match_override
    clip_name = clip_override
    if match_name is None or clip_name is None:
        base = osp.basename(predictions_csv)
        parsed_match, parsed_clip = _parse_match_clip_from_csv_basename(base)
        if match_name is None:
            match_name = parsed_match
        if clip_name is None:
            clip_name = parsed_clip

    if not match_name or not clip_name:
        raise ValueError(
            "无法从 CSV 文件名解析 match/clip，请使用 --match 和 --clip 显式指定。"
            "CSV 文件名需为 xxx_yyy_predictions.csv 形式，或通过参数传入。"
        )

    # 原始帧目录：与项目里数据集约定一致（root_dir / match / clip）
    frame_dir = osp.join(dataset_root, match_name, clip_name)
    if not osp.isdir(frame_dir):
        raise FileNotFoundError(f"帧目录不存在: {frame_dir}")

    half = patch_size // 2
    rows = []
    patch_index = 0

    for idx, row in df.iterrows():
        if not _is_valid_row(row):
            continue
        fname = row["file name"]
        x_center = float(row["x-coordinate"])
        y_center = float(row["y-coordinate"])
        score = row.get("score", np.nan)

        img_path = osp.join(frame_dir, fname)
        if not osp.isfile(img_path):
            print(f"警告: 图像不存在，跳过: {img_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"警告: 无法读取图像，跳过: {img_path}")
            continue

        h, w = img.shape[:2]
        x0 = int(round(x_center - half))
        y0 = int(round(y_center - half))
        x1 = x0 + patch_size
        y1 = y0 + patch_size

        # 边界处理：保证裁剪区域在图像内，不足部分用边界的像素填充（或黑边，这里用复制边界）
        pad_left = max(0, -x0)
        pad_top = max(0, -y0)
        pad_right = max(0, x1 - w)
        pad_bottom = max(0, y1 - h)
        x0_clip = max(0, x0)
        y0_clip = max(0, y0)
        x1_clip = min(w, x1)
        y1_clip = min(h, y1)

        patch = img[y0_clip:y1_clip, x0_clip:x1_clip]
        if pad_left or pad_top or pad_right or pad_bottom:
            patch = cv2.copyMakeBorder(
                patch, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_REPLICATE
            )

        if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
            patch = cv2.resize(patch, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)

        patch_id = f"{match_name}_{clip_name}_{osp.splitext(fname)[0]}_x{int(x_center)}_y{int(y_center)}"
        patch_fname = f"{patch_id}.png"
        patch_path = osp.join(output_dir, patch_fname)
        cv2.imwrite(patch_path, patch)

        rows.append({
            "patch_id": patch_id,
            "source_file": fname,
            "x": x_center,
            "y": y_center,
            "match": match_name,
            "clip": clip_name,
            "patch_path": patch_path,
            "score": score,
            "label": "",  # 预留：标注时填 1=球(TP)，0=非球(FP)
        })
        patch_index += 1

    manifest_path = osp.join(output_dir, "manifest.csv")
    manifest_df = pd.DataFrame(rows)
    manifest_df.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    print(f"已提取 {len(rows)} 个 patch，manifest 已保存: {manifest_path}")
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description="从检测结果 CSV 提取 patch 并生成 manifest")
    parser.add_argument("predictions_csv", help="预测结果 CSV，如 match1_clip1_predictions.csv")
    parser.add_argument("--dataset-root", "-d", required=True,
                        help="原始数据集根目录（如 .../datasets/tennis_predict）")
    parser.add_argument("--output-dir", "-o", default="./patches",
                        help="patch 与 manifest 输出目录，默认 ./patches")
    parser.add_argument("--patch-size", "-p", type=int, default=DEFAULT_PATCH_SIZE,
                        help=f"patch 边长（像素），默认 {DEFAULT_PATCH_SIZE}")
    parser.add_argument("--match", "-m", default=None,
                        help="覆盖从 CSV 文件名解析的 match 名")
    parser.add_argument("--clip", "-c", default=None,
                        help="覆盖从 CSV 文件名解析的 clip 名")
    args = parser.parse_args()

    extract_patches(
        predictions_csv=args.predictions_csv,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        patch_size=args.patch_size,
        match_override=args.match,
        clip_override=args.clip,
    )


if __name__ == "__main__":
    main()
