"""
将 FP 过滤后的检测 CSV（坐标基于裁剪图）转换为 YOLO 格式标签（坐标基于原图）。

背景说明
--------
WASB 模型是在裁剪后的图片上做推理的，因此 CSV 中的 x-coordinate、y-coordinate
是相对于裁剪图的绝对像素坐标。
本脚本通过加上裁剪框的偏移量，将坐标映射回原图坐标系，再归一化为标准 YOLO 格式，
使得生成的标签可以直接用于在原图尺寸的输入数据上（1920×1080）上进行 YOLO 训练或推理。

坐标转换公式
-----------
    x_orig_px = x_crop + crop_left          # 映射到原图 X 像素坐标
    y_orig_px = y_crop + crop_top           # 映射到原图 Y 像素坐标
    x_norm    = x_orig_px / orig_w          # 归一化到 [0, 1]
    y_norm    = y_orig_px / orig_h
    w_norm    = box_size  / orig_w          # 固定框宽归一化
    h_norm    = box_size  / orig_h          # 固定框高归一化

使用示例（在项目根目录执行）
----------------------------
# 基本用法（使用默认参数，与 crop_frames.py 的默认裁剪区域一致）
python fp_filter/csv_to_original_yolo.py ^
    --csv fp_filter/patch_outputs/patches_prediction/match1_clip1_predictions_filtered.csv ^
    --image-root datasets/tennis_predict ^
    --output-dir fp_filter/patch_outputs/patches_prediction/match1_clip1_orig_yolo_labels

# 自定义裁剪偏移和原图尺寸
python fp_filter/csv_to_original_yolo.py ^
    --csv fp_filter/patch_outputs/patches_prediction/match1_clip1_predictions_filtered.csv ^
    --image-root datasets/tennis_predict ^
    --output-dir fp_filter/patch_outputs/patches_prediction/match1_clip1_orig_yolo_labels ^
    --crop-left 650 --crop-top 51 ^
    --orig-w 1920 --orig-h 1080 ^
    --box-size 15 --class-id 0

# 不生成空帧 txt
python fp_filter/csv_to_original_yolo.py ^
    --csv ... --image-root ... --output-dir ... ^
    --no-save-empty
"""

import argparse
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def parse_match_clip(csv_path: Path):
    """从 CSV 文件名解析 match 和 clip。
    支持 match1_clip1_predictions_filtered.csv 或 match1_clip1_predictions.csv。
    """
    name = csv_path.name
    if name.endswith("_predictions_filtered.csv"):
        core = name[: -len("_predictions_filtered.csv")]
    elif name.endswith("_predictions.csv"):
        core = name[: -len("_predictions.csv")]
    else:
        return None, None
    parts = core.split("_")
    if len(parts) >= 2:
        return parts[0], parts[1]
    return None, None


def is_valid_detection(row) -> bool:
    """判断该行是否为 visibility=1 且坐标为有限数值。"""
    try:
        vis = int(row.get("visibility", 0))
    except (TypeError, ValueError):
        return False
    if vis != 1:
        return False

    x = row.get("x-coordinate")
    y = row.get("y-coordinate")
    if pd.isna(x) or pd.isna(y):
        return False
    try:
        xf, yf = float(x), float(y)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(xf) or not np.isfinite(yf):
        return False
    return True


def collect_all_frame_names(image_root: Path, match_name: str, clip_name: str) -> list:
    """
    扫描 image_root/match/clip 下的所有 .jpg 文件名（不含路径），
    用于确保每一个原图帧都有对应的 txt（空帧写空文件）。

    若目录不存在，返回空列表（脚本仍会处理 CSV 中出现的帧）。
    """
    if not match_name or not clip_name:
        return []
    frame_dir = image_root / match_name / clip_name
    if not frame_dir.exists():
        print(f"[提示] 帧目录不存在，跳过全量帧扫描: {frame_dir}")
        return []
    names = sorted(p.name for p in frame_dir.glob("*.jpg"))
    return names


# ─────────────────────────────────────────────────────────────────────────────
# 核心转换逻辑
# ─────────────────────────────────────────────────────────────────────────────

def convert_csv_to_original_yolo(
    csv_path: Path,
    image_root: Path,
    output_dir: Path,
    crop_left: int,
    crop_top: int,
    orig_w: int,
    orig_h: int,
    box_size: float,
    class_id: int,
    save_empty: bool,
):
    """
    读取 CSV（坐标基于裁剪图），将坐标转换为基于原图的 YOLO 归一化格式，
    并写入 output_dir 下的逐帧 .txt 文件。

    Args:
        csv_path   : 过滤后的 CSV 文件路径
        image_root : 原图所在根目录（用于获取全量帧文件名）
        output_dir : YOLO txt 输出目录
        crop_left  : 裁剪框左边界在原图中的 X 坐标（像素）
        crop_top   : 裁剪框上边界在原图中的 Y 坐标（像素）
        orig_w     : 原图宽度（像素）
        orig_h     : 原图高度（像素）
        box_size   : 固定检测框的像素边长（宽高相等）
        class_id   : YOLO 类别 ID
        save_empty : 是否为无检测帧写入空 txt
    """
    print(f"读取 CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    if "file name" not in df.columns:
        raise ValueError(f"CSV 缺少 'file name' 列，请确认文件格式: {csv_path}")

    match_name, clip_name = parse_match_clip(csv_path)
    print(f"解析到: match={match_name}, clip={clip_name}")

    # ── 计算原图归一化的固定框宽高（box_size 固定，不受裁剪区影响）
    w_norm = box_size / orig_w
    h_norm = box_size / orig_h

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 按帧分组处理 CSV
    grouped = df.groupby("file name", sort=True)

    # ── 获取全量帧文件名集合，用于补全空帧 txt
    all_frames = set()
    if save_empty:
        all_frames = set(collect_all_frame_names(image_root, match_name, clip_name))
        if all_frames:
            print(f"共扫描到原图帧: {len(all_frames)} 张")

    # ── 统计计数器
    total_frames_in_csv = 0
    frames_with_boxes = 0
    total_boxes = 0
    discarded_boxes = 0  # 转换后坐标越界（理论上不应发生）被丢弃的框数

    # ── 逐帧转换并写入
    written_stems = set()  # 已写入的帧 stem，用于后续补全空帧

    for fname, frame_df in grouped:
        total_frames_in_csv += 1
        stem = Path(str(fname)).stem
        txt_path = output_dir / f"{stem}.txt"
        written_stems.add(stem)

        valid_rows = [row for _, row in frame_df.iterrows() if is_valid_detection(row)]

        if len(valid_rows) == 0:
            # 无有效检测：写空文件或跳过
            if save_empty:
                txt_path.write_text("", encoding="utf-8")
            continue

        lines = []
        for row in valid_rows:
            # ── Step 1: 读取裁剪图上的绝对像素坐标
            x_crop = float(row["x-coordinate"])
            y_crop = float(row["y-coordinate"])

            # ── Step 2: 加上裁剪偏移，映射到原图绝对像素坐标
            x_orig_px = x_crop + crop_left
            y_orig_px = y_crop + crop_top

            # ── Step 3: 归一化到原图 [0, 1]
            x_norm = x_orig_px / orig_w
            y_norm = y_orig_px / orig_h

            # ── Step 4: 边界检查（严格模式：越界则丢弃该检测点）
            # 中心点越界说明裁剪图坐标超出了裁剪区域，属于异常数据
            if not (0.0 <= x_norm <= 1.0 and 0.0 <= y_norm <= 1.0):
                discarded_boxes += 1
                print(
                    f"  [警告] 帧 {fname}: 转换后坐标越界 "
                    f"(x_crop={x_crop:.1f}, y_crop={y_crop:.1f}) "
                    f"-> (x_norm={x_norm:.4f}, y_norm={y_norm:.4f})，已丢弃"
                )
                continue

            lines.append(
                f"{class_id} {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}"
            )
            total_boxes += 1

        if lines:
            txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            frames_with_boxes += 1
        elif save_empty:
            # 有效行全部被丢弃（越界），按空帧处理
            txt_path.write_text("", encoding="utf-8")

    # ── 补全全量帧中在 CSV 里未出现的帧（纯空帧，无任何检测）
    empty_supplemented = 0
    if save_empty and all_frames:
        for frame_fname in all_frames:
            stem = Path(frame_fname).stem
            if stem not in written_stems:
                txt_path = output_dir / f"{stem}.txt"
                txt_path.write_text("", encoding="utf-8")
                empty_supplemented += 1

    # ── 打印统计摘要
    print("\n" + "=" * 60)
    print("转换完成，统计摘要：")
    print(f"  CSV 中帧数          : {total_frames_in_csv}")
    print(f"  有检测框的帧数      : {frames_with_boxes}")
    print(f"  写入的检测框总数    : {total_boxes}")
    if discarded_boxes > 0:
        print(f"  ⚠ 丢弃（越界）框数 : {discarded_boxes}")
    if save_empty:
        print(f"  补全的空帧 txt 数   : {empty_supplemented}")
    print(f"  输出目录            : {output_dir}")
    print("=" * 60)
    print()
    print("坐标转换参数：")
    print(f"  裁剪图坐标偏移  : crop_left={crop_left}, crop_top={crop_top}")
    print(f"  原图尺寸        : {orig_w} x {orig_h}")
    print(f"  固定框大小      : {box_size} px  →  w_norm={w_norm:.6f}, h_norm={h_norm:.6f}")
    print()
    print("验证示例（检查第一个有效框）：")
    first = df[df.get("visibility", 0) == 1] if "visibility" in df.columns else df.head(1)
    if not first.empty:
        r = first.iloc[0]
        try:
            xc = float(r["x-coordinate"])
            yc = float(r["y-coordinate"])
            xa = xc + crop_left
            ya = yc + crop_top
            print(f"  裁剪图坐标 ({xc:.1f}, {yc:.1f})")
            print(f"  原图像素坐标 ({xa:.1f}, {ya:.1f})")
            print(f"  YOLO 归一化 ({xa/orig_w:.6f}, {ya/orig_h:.6f})")
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# 命令行入口
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="将 FP 过滤后的 CSV（坐标基于裁剪图）转换为原图尺度的 YOLO txt 标签",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
裁剪参数说明（与 crop_frames.py 的默认值一致）：
  --crop-left  650   裁剪框左边界在原图的 X 坐标
  --crop-top   51    裁剪框上边界在原图的 Y 坐标
  默认原图尺寸 1920x1080，与项目约定一致。

输出 txt 格式（标准 YOLO）：
  class x_center y_center width height
  （所有值均归一化到原图尺寸 [0, 1]）
        """
    )
    parser.add_argument(
        "--csv", "-c",
        type=str,
        required=True,
        help="过滤后的检测 CSV 文件路径（如 match1_clip1_predictions_filtered.csv）",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default="",
        help="原图所在根目录（如 datasets/tennis_predict）。"
             "用于扫描全量帧文件名以补全空帧 txt。留空则只处理 CSV 中出现的帧。",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="YOLO txt 输出目录。默认在 CSV 同目录下生成 <csv_stem>_orig_yolo_labels 文件夹。",
    )
    # ── 裁剪偏移参数（与 crop_frames.py 默认值保持一致）
    parser.add_argument(
        "--crop-left",
        type=int,
        default=650,
        help="裁剪框左边界在原图中的 X 坐标（像素，默认: 650）",
    )
    parser.add_argument(
        "--crop-top",
        type=int,
        default=51,
        help="裁剪框上边界在原图中的 Y 坐标（像素，默认: 51）",
    )
    # ── 原图尺寸
    parser.add_argument(
        "--orig-w",
        type=int,
        default=1920,
        help="原图宽度（像素，默认: 1920）",
    )
    parser.add_argument(
        "--orig-h",
        type=int,
        default=1080,
        help="原图高度（像素，默认: 1080）",
    )
    # ── YOLO 标签参数
    parser.add_argument(
        "--box-size",
        type=float,
        default=15.0,
        help="固定检测框的像素边长（默认: 15）",
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=0,
        help="YOLO 类别 ID（默认: 0）",
    )
    parser.add_argument(
        "--no-save-empty",
        action="store_true",
        default=False,
        help="如果设置，则不为无检测的帧生成空 txt 文件（默认：生成空 txt）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        print(f"错误: CSV 文件不存在: {csv_path}")
        return

    image_root = Path(args.image_root) if args.image_root else Path("")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # 默认：CSV 同级目录下，以 CSV stem 命名的文件夹
        output_dir = csv_path.parent / f"{csv_path.stem}_orig_yolo_labels"

    print("=" * 60)
    print("FP 过滤 CSV → 原图尺度 YOLO 标签 转换工具")
    print("=" * 60)
    print(f"输入 CSV     : {csv_path}")
    print(f"原图根目录   : {image_root if image_root.parts else '（未指定）'}")
    print(f"输出目录     : {output_dir}")
    print(f"裁剪偏移     : left={args.crop_left}, top={args.crop_top}")
    print(f"原图尺寸     : {args.orig_w} x {args.orig_h}")
    print(f"固定框大小   : {args.box_size} px")
    print(f"YOLO 类别 ID : {args.class_id}")
    print(f"生成空帧 txt : {'否' if args.no_save_empty else '是'}")
    print()

    convert_csv_to_original_yolo(
        csv_path=csv_path,
        image_root=image_root,
        output_dir=output_dir,
        crop_left=args.crop_left,
        crop_top=args.crop_top,
        orig_w=args.orig_w,
        orig_h=args.orig_h,
        box_size=args.box_size,
        class_id=args.class_id,
        save_empty=not args.no_save_empty,
    )


if __name__ == "__main__":
    main()
