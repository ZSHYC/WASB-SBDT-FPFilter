"""
organize_comparisons.py

根据 reports/detection_comparison.csv 将 only_wasb / only_yolo / both 三类图片及标签
组织到指定输出目录。脚本会：

- 读取比较结果 CSV（默认路径：reports/detection_comparison.csv）
- 对于 wasb：从 wasb_outputs/<dataset>/ 中寻找对应的 comparison images (game 目录)
  和 predictions CSV（*_predictions.csv），读取原始球心坐标 (像素坐标)
- 对于 yolo：从 yolo_root/<dataset>/<game>/comparison/ 读取比较图像，从
  yolo_root/<dataset>/<game>/labels/ 读取 yolo 格式标签（*.txt）
- 将三类图片分别存入输出目录：only_wasb / only_yolo / yolo_and_wasb
  每个输出目录下包含 `images/` 和 `labels/`（wasb 还包含 `labels_origin/`）

标签处理说明：
- wasb 原始 labels 是球心 (x,y) 的像素坐标，脚本会把原始坐标保存到
  `labels_origin`（每张图片一个 txt，内容：x y），并基于该球心构造一个
  YOLO 风格的 bbox（class_id x_center y_center w h，均为归一化），
  其宽高通过下述方式估算：优先使用同 game 的 YOLO 输出标签中 bbox 的
  平均宽高；如果不存在则使用默认宽高（可通过参数调整）。

如何运行（示例）：
python src/tools/organize_comparisons.py \
  --comparison-csv "reports/detection_comparison.csv" \
  --wasb-root "D:/Personal/Desktop/WASB-SBDT-FPFilter/wasb_outputs" \
  --yolo-root "D:/Personal/Desktop/yolo/yolo_output" \
  --out-root "D:/Personal/Desktop/WASB-SBDT-FPFilter/organized_outputs"
  
使用：
python src/tools/organize_comparisons.py --comparison-csv "reports/detection_comparison.csv" --wasb-root "D:/Personal/Desktop/WASB-SBDT-FPFilter/wasb_outputs" --yolo-root "D:/Personal/Desktop/yolo/yolo_output" --out-root "D:/Personal/Desktop/WASB-SBDT-FPFilter/organized_outputs"  

注意：脚本只会在源文件存在时复制对应的比较图；若某一方的比较图找不到，
会根据用户要求跳过（即不强制从另一位置生成或替代）。

"""
import argparse
import csv
import glob
import math
import os
import shutil
from typing import Dict, Tuple, Optional, List


IMG_W = 1920
IMG_H = 1080
DEFAULT_BOX_W = 0.02
DEFAULT_BOX_H = 0.02


def safe_makedirs(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def find_wasb_game_dir(wasb_dataset_dir: str, game: str) -> Optional[str]:
    # 寻找以 game 名称开头的文件夹（例如 game_1 -> game_1_Clip1）
    if not os.path.isdir(wasb_dataset_dir):
        return None
    for name in os.listdir(wasb_dataset_dir):
        full = os.path.join(wasb_dataset_dir, name)
        if os.path.isdir(full) and name.startswith(game + "_"):
            return full
    return None


def find_wasb_predictions_csv(wasb_dataset_dir: str, game: str) -> Optional[str]:
    pattern = os.path.join(wasb_dataset_dir, f"{game}_*predictions.csv")
    matches = glob.glob(pattern)
    return matches[0] if matches else None


def read_wasb_predictions(pred_csv_path: str) -> Dict[str, Tuple[float, float]]:
    # 返回 map: filename_without_ext (e.g. img_1) -> (x,y)（像素坐标）
    d = {}
    if not os.path.exists(pred_csv_path):
        return d
    with open(pred_csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row.get('file name') or row.get('file_name') or row.get('filename')
            if not fname:
                continue
            key = os.path.splitext(fname)[0]
            try:
                x = float(row.get('x-coordinate', 'nan'))
                y = float(row.get('y-coordinate', 'nan'))
            except Exception:
                x = float('nan')
                y = float('nan')
            d[key] = (x, y)
    return d


def read_yolo_label_file(label_path: str) -> List[Tuple[int, float, float, float, float]]:
    # 每行：class x y w h（均是归一化）
    res = []
    if not os.path.exists(label_path):
        return res
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cls = int(parts[0])
                x, y, w, h = map(float, parts[1:5])
                res.append((cls, x, y, w, h))
            except Exception:
                continue
    return res


def estimate_avg_wh_from_yolo(yolo_labels_dir: str) -> Tuple[float, float]:
    # 统计目录中所有 label 文件的 w,h 平均值（归一化）
    if not os.path.isdir(yolo_labels_dir):
        return DEFAULT_BOX_W, DEFAULT_BOX_H
    ws = []
    hs = []
    for p in glob.glob(os.path.join(yolo_labels_dir, "*.txt")):
        for _cls, _x, _y, w, h in read_yolo_label_file(p):
            ws.append(w)
            hs.append(h)
    if not ws:
        return DEFAULT_BOX_W, DEFAULT_BOX_H
    return float(sum(ws)) / len(ws), float(sum(hs)) / len(hs)


def copy_file_if_exists(src: str, dst: str) -> bool:
    if src and os.path.exists(src):
        safe_makedirs(os.path.dirname(dst))
        shutil.copy2(src, dst)
        return True
    return False


def write_yolo_label(path: str, class_id: int, x: float, y: float, w: float, h: float):
    safe_makedirs(os.path.dirname(path))
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def write_origin_label(path: str, x: float, y: float):
    safe_makedirs(os.path.dirname(path))
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f"{x} {y}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--comparison-csv', default='reports/detection_comparison.csv')
    parser.add_argument('--wasb-root', default=os.path.join(os.getcwd(), 'wasb_outputs'))
    parser.add_argument('--yolo-root', default='D:/Personal/Desktop/yolo/yolo_output')
    parser.add_argument('--out-root', default=os.path.join(os.getcwd(), 'organized_outputs'))
    parser.add_argument('--img-w', type=int, default=IMG_W)
    parser.add_argument('--img-h', type=int, default=IMG_H)
    parser.add_argument('--class-id', type=int, default=0, help='class id to write for wasb-generated bboxes')
    parser.add_argument('--default-w', type=float, default=DEFAULT_BOX_W)
    parser.add_argument('--default-h', type=float, default=DEFAULT_BOX_H)
    args = parser.parse_args()

    IMG_W_LOCAL = args.img_w
    IMG_H_LOCAL = args.img_h

    import pandas as pd
    df = pd.read_csv(args.comparison_csv)

    # 准备 output 根目录与子目录
    only_wasb_images = os.path.join(args.out_root, 'only_wasb', 'images')
    only_wasb_labels = os.path.join(args.out_root, 'only_wasb', 'labels')
    only_wasb_labels_origin = os.path.join(args.out_root, 'only_wasb', 'labels_origin')

    only_yolo_images = os.path.join(args.out_root, 'only_yolo', 'images')
    only_yolo_labels = os.path.join(args.out_root, 'only_yolo', 'labels')

    both_images = os.path.join(args.out_root, 'yolo_and_wasb', 'images')
    both_labels = os.path.join(args.out_root, 'yolo_and_wasb', 'labels')

    # group by dataset and game for efficiency
    grouped = df.groupby(['dataset', 'game'])

    for (dataset, game), sub in grouped:
        dataset_str = str(dataset)
        print(f"Processing dataset={dataset_str} game={game} rows={len(sub)}")

        wasb_dataset_dir = os.path.join(args.wasb_root, dataset_str)
        wasb_game_dir = find_wasb_game_dir(wasb_dataset_dir, game)
        wasb_pred_csv = find_wasb_predictions_csv(wasb_dataset_dir, game)
        wasb_preds = read_wasb_predictions(wasb_pred_csv) if wasb_pred_csv else {}

        yolo_game_dir = os.path.join(args.yolo_root, dataset_str, game)
        yolo_labels_dir = os.path.join(yolo_game_dir, 'labels')
        yolo_comp_dir = os.path.join(yolo_game_dir, 'comparison')

        # 估计 bbox 大小（优先使用 yolo 提供的标签统计）
        avg_w, avg_h = estimate_avg_wh_from_yolo(yolo_labels_dir)
        # 如果 yolo 没有则使用传入的默认
        if avg_w == DEFAULT_BOX_W and avg_h == DEFAULT_BOX_H:
            avg_w = args.default_w
            avg_h = args.default_h

        for _, row in sub.iterrows():
            img_key = row['image']  # e.g. img_1
            img_fname = img_key + '.jpg'
            status = row.get('status')

            # Paths for wasb comparison image
            wasb_img_src = None
            if wasb_game_dir:
                candidate = os.path.join(wasb_game_dir, img_fname)
                if os.path.exists(candidate):
                    wasb_img_src = candidate

            # Paths for yolo comparison image
            yolo_img_src = os.path.join(yolo_comp_dir, img_fname)
            if not os.path.exists(yolo_img_src):
                yolo_img_src = None

            # For yolo label source file
            yolo_label_src = os.path.join(yolo_labels_dir, img_key + '.txt')
            if not os.path.exists(yolo_label_src):
                yolo_label_src = None

            if status == 'only_wasb':
                # Copy wasb comparison image (if exists)
                if wasb_img_src:
                    dst_img = os.path.join(only_wasb_images, dataset_str, game, img_fname)
                    copy_file_if_exists(wasb_img_src, dst_img)
                else:
                    print(f"  wasb image not found for {dataset_str}/{game}/{img_fname}, skipped image")

                # write origin label and generated yolo label
                pred = wasb_preds.get(img_key)
                if pred:
                    x_pix, y_pix = pred
                    # skip invalid coordinates
                    if math.isfinite(x_pix) and math.isfinite(y_pix):
                        # origin
                        origin_dst = os.path.join(only_wasb_labels_origin, dataset_str, game, img_key + '.txt')
                        write_origin_label(origin_dst, x_pix, y_pix)

                        # normalized center
                        x_norm = x_pix / IMG_W_LOCAL
                        y_norm = y_pix / IMG_H_LOCAL
                        w_norm = avg_w
                        h_norm = avg_h
                        label_dst = os.path.join(only_wasb_labels, dataset_str, game, img_key + '.txt')
                        write_yolo_label(label_dst, args.class_id, x_norm, y_norm, w_norm, h_norm)
                    else:
                        print(f"  wasb pred invalid for {img_key}, skipping label")
                else:
                    print(f"  wasb prediction not found in CSV for {img_key}")

            elif status == 'only_yolo':
                # Copy yolo comparison image if exists (user said: 找不到就跳过，不要了)
                if yolo_img_src:
                    dst_img = os.path.join(only_yolo_images, dataset_str, game, img_fname)
                    copy_file_if_exists(yolo_img_src, dst_img)
                else:
                    print(f"  yolo comparison image not found for {dataset_str}/{game}/{img_fname}, skipping image")

                # copy yolo label if exists
                if yolo_label_src:
                    dst_label = os.path.join(only_yolo_labels, dataset_str, game, img_key + '.txt')
                    copy_file_if_exists(yolo_label_src, dst_label)
                else:
                    print(f"  yolo label not found for {img_key}, skipping label")

            elif status == 'both':
                # Prefer yolo comparison image when available, else wasb
                src_img = yolo_img_src or wasb_img_src
                if src_img:
                    dst_img = os.path.join(both_images, dataset_str, game, img_fname)
                    copy_file_if_exists(src_img, dst_img)
                else:
                    print(f"  neither yolo nor wasb comparison image found for {dataset_str}/{game}/{img_fname}")

                # For labels we use YOLO labels if available (as user requested labels 为 yolo 模型输出)
                if yolo_label_src:
                    dst_label = os.path.join(both_labels, dataset_str, game, img_key + '.txt')
                    copy_file_if_exists(yolo_label_src, dst_label)
                else:
                    # fall back: if yolo label missing but wasb has pred -> generate
                    pred = wasb_preds.get(img_key)
                    if pred and math.isfinite(pred[0]) and math.isfinite(pred[1]):
                        x_pix, y_pix = pred
                        x_norm = x_pix / IMG_W_LOCAL
                        y_norm = y_pix / IMG_H_LOCAL
                        label_dst = os.path.join(both_labels, dataset_str, game, img_key + '.txt')
                        write_yolo_label(label_dst, args.class_id, x_norm, y_norm, avg_w, avg_h)
                        print(f"  generated fallback yolo label for {img_key} from wasb pred")
                    else:
                        print(f"  no yolo label and no valid wasb pred for {img_key}")

            else:
                # neither or unknown status: ignore
                continue

    print('Done.')


if __name__ == '__main__':
    main()
