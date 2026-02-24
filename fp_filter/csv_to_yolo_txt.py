import argparse
import math
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def parse_match_clip(csv_path: Path):
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


def is_valid_detection(row):
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
        xf = float(x)
        yf = float(y)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(xf) or not np.isfinite(yf):
        return False
    return True


def build_candidate_paths(fname: str, image_root: Path, match_name: str, clip_name: str):
    candidates = [image_root / fname]
    if match_name and clip_name:
        candidates.append(image_root / match_name / clip_name / fname)
        candidates.append(image_root / f"{match_name}_{clip_name}" / fname)
    return candidates


def resolve_image_size(
    fname: str,
    csv_path: Path,
    image_root: Path,
    match_name: str,
    clip_name: str,
    path_cache: dict,
    size_cache: dict,
):
    if fname in size_cache:
        return size_cache[fname]

    if fname in path_cache:
        img = cv2.imread(str(path_cache[fname]))
        if img is not None:
            h, w = img.shape[:2]
            size_cache[fname] = (w, h)
            return w, h

    search_roots = []
    if image_root is not None:
        search_roots.append(image_root)
    search_roots.extend([csv_path.parent, csv_path.parent.parent, Path.cwd()])

    seen = set()
    unique_roots = []
    for root in search_roots:
        if root is None:
            continue
        rp = root.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        unique_roots.append(root)

    for root in unique_roots:
        for p in build_candidate_paths(fname, root, match_name, clip_name):
            if p.is_file():
                img = cv2.imread(str(p))
                if img is not None:
                    h, w = img.shape[:2]
                    path_cache[fname] = p
                    size_cache[fname] = (w, h)
                    return w, h

    if image_root is not None and image_root.exists():
        hits = list(image_root.rglob(fname))
        if hits:
            p = hits[0]
            img = cv2.imread(str(p))
            if img is not None:
                h, w = img.shape[:2]
                path_cache[fname] = p
                size_cache[fname] = (w, h)
                return w, h

    return None


def infer_fallback_size(df: pd.DataFrame):
    valid = df.copy()
    valid = valid[pd.to_numeric(valid["x-coordinate"], errors="coerce").notna()]
    valid = valid[pd.to_numeric(valid["y-coordinate"], errors="coerce").notna()]
    if len(valid) == 0:
        return 512, 288

    x = pd.to_numeric(valid["x-coordinate"], errors="coerce").astype(float)
    y = pd.to_numeric(valid["y-coordinate"], errors="coerce").astype(float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return 512, 288

    w = int(math.ceil(float(x.max()) + 16))
    h = int(math.ceil(float(y.max()) + 16))
    w = max(w, 64)
    h = max(h, 64)
    return w, h


def write_yolo_labels(
    csv_path: Path,
    output_dir: Path,
    image_root: Path,
    box_size: float,
    class_id: int,
    save_empty: bool,
):
    df = pd.read_csv(csv_path)
    if "file name" not in df.columns:
        raise ValueError("CSV 缺少 'file name' 列")

    match_name, clip_name = parse_match_clip(csv_path)
    fallback_w, fallback_h = infer_fallback_size(df)

    output_dir.mkdir(parents=True, exist_ok=True)

    grouped = df.groupby("file name", sort=True)
    path_cache = {}
    size_cache = {}

    total_frames = 0
    frames_with_boxes = 0
    total_boxes = 0
    unresolved_size_frames = 0

    for fname, frame_df in grouped:
        total_frames += 1
        stem = Path(str(fname)).stem
        txt_path = output_dir / f"{stem}.txt"

        valid_rows = [row for _, row in frame_df.iterrows() if is_valid_detection(row)]

        if len(valid_rows) == 0:
            if save_empty:
                txt_path.write_text("", encoding="utf-8")
            continue

        size = resolve_image_size(
            fname=str(fname),
            csv_path=csv_path,
            image_root=image_root,
            match_name=match_name,
            clip_name=clip_name,
            path_cache=path_cache,
            size_cache=size_cache,
        )

        if size is None:
            img_w, img_h = fallback_w, fallback_h
            unresolved_size_frames += 1
        else:
            img_w, img_h = size

        box_w = min(float(box_size), float(img_w))
        box_h = min(float(box_size), float(img_h))
        w_norm = box_w / float(img_w)
        h_norm = box_h / float(img_h)

        lines = []
        for row in valid_rows:
            xc = float(row["x-coordinate"]) / float(img_w)
            yc = float(row["y-coordinate"]) / float(img_h)
            xc = min(max(xc, 0.0), 1.0)
            yc = min(max(yc, 0.0), 1.0)
            lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w_norm:.6f} {h_norm:.6f}")

        txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        frames_with_boxes += 1
        total_boxes += len(lines)

    print(f"输入 CSV: {csv_path}")
    print(f"输出目录: {output_dir}")
    print(f"总帧数: {total_frames}")
    print(f"有框帧数: {frames_with_boxes}")
    print(f"总框数: {total_boxes}")
    print(f"未能定位图片尺寸的帧数: {unresolved_size_frames}")
    if unresolved_size_frames > 0:
        print(f"已使用后备尺寸: {fallback_w}x{fallback_h}")


def main():
    parser = argparse.ArgumentParser(description="将推理后的 CSV 转成逐帧 YOLO txt 标签")
    parser.add_argument("--csv", required=True, help="推理后的 CSV 路径")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="输出 txt 目录，默认与 CSV 同级: <csv_stem>_yolo_labels",
    )
    parser.add_argument(
        "--image-root",
        default=None,
        help="图像根目录（可选）。程序会优先在这里自动查找每帧图片并读取尺寸",
    )
    parser.add_argument("--box-size", type=float, default=15.0, help="固定框大小（像素），默认 15")
    parser.add_argument("--class-id", type=int, default=0, help="YOLO 类别 ID，默认 0")
    parser.add_argument(
        "--save-empty",
        action="store_true",
        help="对没有有效检测的帧也写空 txt（默认开启）",
    )
    parser.add_argument(
        "--no-save-empty",
        action="store_true",
        help="不为无有效检测帧创建空 txt",
    )

    args = parser.parse_args()

    csv_path = Path(args.csv).resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV 不存在: {csv_path}")

    if args.output_dir is None:
        output_dir = csv_path.parent / f"{csv_path.stem}_yolo_labels"
    else:
        output_dir = Path(args.output_dir).resolve()

    image_root = Path(args.image_root).resolve() if args.image_root else None

    save_empty = True
    if args.no_save_empty:
        save_empty = False
    elif args.save_empty:
        save_empty = True

    write_yolo_labels(
        csv_path=csv_path,
        output_dir=output_dir,
        image_root=image_root,
        box_size=args.box_size,
        class_id=args.class_id,
        save_empty=save_empty,
    )


if __name__ == "__main__":
    main()
