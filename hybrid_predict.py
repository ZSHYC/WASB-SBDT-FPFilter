"""
使用示例（包含可视化，推荐）
--------
# 推荐：运行并同时生成可视化图片与视频（生产/调试常用）
python hybrid_predict.py ^
    --input-folder datasets/tennis_predict/match1/clip1_yolo ^
    --output-folder hybrid_outputs/match1_clip1_labels ^
    --yolo-model yolov8n_1280_1113.pt ^
    --wasb-labels-dir fp_filter/patch_outputs/patches_prediction/match1_clip1_orig_yolo_labels ^
    --conf 0.5 ^
    --orig-w 1920 --orig-h 1080 ^
    --left 650 --top 51 --right 1236 --bottom 339 ^
    --nms-iou 0.0 ^
    --visualize --visualize-video

# 简洁示例（使用 pipeline 输出的原图尺度 WASB 标签目录，并生成可视化）
python hybrid_predict.py --input-folder clip1_yolo --output-folder hybrid_outputs5/match1_clip1_labels --yolo-model yolov8n_1280_1113.pt --wasb-labels-dir pipeline_outputs/2026-02-25_14-25-25/stage5_original_yolo_labels/match1_clip1_predictions_filtered_orig_yolo_labels --visualize --visualize-video

3. 本脚本对原图跑 YOLO，并过滤 ROI 内检测，再与 --wasb-labels-dir 合并输出。

默认行为：脚本会生成可视化图片与视频（需安装 `opencv-python`）。如需关闭可使用 `--no-visualize` 或 `--no-visualize-video`。

"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

from ultralytics import YOLO
 

Detection = Dict[str, float]


def parse_args():
    parser = argparse.ArgumentParser(
        description="融合推理：ROI 内 WASB+FP_Filter，ROI 外 YOLO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-folder", type=str, required=True, help="原图文件夹（1920x1080）")
    parser.add_argument("--output-folder", type=str, required=True, help="融合后 YOLO txt 输出目录")
    parser.add_argument("--yolo-model", type=str, required=True, help="YOLO 模型权重路径")
    parser.add_argument("--wasb-labels-dir", type=str, required=True, help="WASB+FP 转换后的原图尺度 YOLO 标签目录")
    parser.add_argument("--conf", type=float, default=0.5, help="YOLO 置信度阈值")

    parser.add_argument("--orig-w", type=int, default=1920, help="原图宽")
    parser.add_argument("--orig-h", type=int, default=1080, help="原图高")
    parser.add_argument("--left", type=int, default=650, help="ROI 左边界")
    parser.add_argument("--top", type=int, default=51, help="ROI 上边界")
    parser.add_argument("--right", type=int, default=1236, help="ROI 右边界")
    parser.add_argument("--bottom", type=int, default=339, help="ROI 下边界")


    parser.add_argument(
        "--inside-policy",
        type=str,
        default="center",
        choices=["center", "any_overlap"],
        help="判定 YOLO 框是否属于 ROI 的规则：center=中心点在 ROI 内；any_overlap=与 ROI 有重叠即视为 ROI 内",
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.0,
        help="融合后按类别执行 NMS 的 IoU 阈值；<=0 表示关闭 NMS",
    )
    parser.add_argument(
        "--no-save-empty",
        action="store_true",
        default=False,
        help="不写空 txt（默认写空 txt）",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="最多处理多少张图（0 表示全部处理，便于快速验证/分批跑）",
    )
    # 默认开启可视化，但提供 --no-visualize / --no-visualize-video 用于显式关闭
    parser.add_argument(
        "--visualize",
        dest="visualize",
        action="store_true",
        default=True,
        help="是否基于融合后的 labels 在原图上绘制可视化图片（默认: 开启）",
    )
    parser.add_argument(
        "--no-visualize",
        dest="visualize",
        action="store_false",
        help="关闭图片可视化输出",
    )
    parser.add_argument(
        "--visualize-video",
        dest="visualize_video",
        action="store_true",
        default=True,
        help="是否基于融合后的 labels 生成可视化视频（默认: 开启）",
    )
    parser.add_argument(
        "--no-visualize-video",
        dest="visualize_video",
        action="store_false",
        help="关闭可视化视频输出",
    )
    parser.add_argument(
        "--visualize-dir",
        type=str,
        default="",
        help="可视化图片输出目录（留空则默认为 <output-folder>_visualized）",
    )
    parser.add_argument(
        "--video-path",
        type=str,
        default="",
        help="可视化视频输出路径（留空则默认为 <output-folder>_visualized.mp4）",
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=25,
        help="可视化视频帧率",
    )
    parser.add_argument(
        "--video-fourcc",
        type=str,
        default="mp4v",
        help="可视化视频编码 fourcc（4 个字符，如 mp4v/XVID）",
    )
    return parser.parse_args()


def validate_args(args):
    if args.orig_w <= 0 or args.orig_h <= 0:
        raise ValueError("orig-w/orig-h 必须 > 0")
    if not (0.0 <= args.conf <= 1.0):
        raise ValueError("conf 必须在 [0,1]")
    if args.left >= args.right or args.top >= args.bottom:
        raise ValueError("ROI 参数无效：需要 left < right 且 top < bottom")
    if args.nms_iou < 0.0 or args.nms_iou >= 1.0:
        raise ValueError("nms-iou 必须满足 0 <= nms-iou < 1")
    if args.max_images < 0:
        raise ValueError("max-images 必须 >= 0")
    if args.video_fps <= 0:
        raise ValueError("video-fps 必须 > 0")
    if len(args.video_fourcc) != 4:
        raise ValueError("video-fourcc 必须是 4 个字符")


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}


def denorm_xywh_to_xyxy(x: float, y: float, w: float, h: float) -> Tuple[float, float, float, float]:
    x1 = x - w / 2.0
    y1 = y - h / 2.0
    x2 = x + w / 2.0
    y2 = y + h / 2.0
    return x1, y1, x2, y2


def iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    if denom <= 0:
        return 0.0
    return inter / denom


def nms_by_class(dets: List[Detection], iou_thr: float) -> List[Detection]:
    if iou_thr <= 0.0 or len(dets) <= 1:
        return dets

    by_cls: Dict[int, List[Detection]] = {}
    for det in dets:
        cls_id = int(det["cls"])
        by_cls.setdefault(cls_id, []).append(det)

    kept: List[Detection] = []
    for cls_id, cls_dets in by_cls.items():
        cls_sorted = sorted(cls_dets, key=lambda d: float(d["conf"]), reverse=True)
        while cls_sorted:
            best = cls_sorted.pop(0)
            kept.append(best)
            best_box = denorm_xywh_to_xyxy(best["x"], best["y"], best["w"], best["h"])
            remain = []
            for d in cls_sorted:
                box = denorm_xywh_to_xyxy(d["x"], d["y"], d["w"], d["h"])
                if iou_xyxy(best_box, box) < iou_thr:
                    remain.append(d)
            cls_sorted = remain
    return kept


def inside_roi(det: Detection, left: int, top: int, right: int, bottom: int, policy: str) -> bool:
    x = float(det["x"])
    y = float(det["y"])
    w = float(det["w"])
    h = float(det["h"])

    if policy == "center":
        return left <= x <= right and top <= y <= bottom

    # any_overlap
    x1, y1, x2, y2 = denorm_xywh_to_xyxy(x, y, w, h)
    roi_x1, roi_y1, roi_x2, roi_y2 = float(left), float(top), float(right), float(bottom)
    inter_x1 = max(x1, roi_x1)
    inter_y1 = max(y1, roi_y1)
    inter_x2 = min(x2, roi_x2)
    inter_y2 = min(y2, roi_y2)
    return inter_x2 > inter_x1 and inter_y2 > inter_y1


def parse_yolo_line(line: str, src: str = "wasb") -> Detection:
    parts = line.strip().split()
    if len(parts) < 5:
        raise ValueError("YOLO 行字段不足 5 列")
    cls_id = int(float(parts[0]))
    x = float(parts[1])
    y = float(parts[2])
    w = float(parts[3])
    h = float(parts[4])
    conf = float(parts[5]) if len(parts) >= 6 else 1.0
    return {"cls": cls_id, "x": x, "y": y, "w": w, "h": h, "conf": conf, "src": src}


def read_wasb_labels_for_stem(wasb_labels_dir: Path, stem: str) -> List[Detection]:
    txt_path = wasb_labels_dir / f"{stem}.txt"
    if not txt_path.exists():
        return []

    detections: List[Detection] = []
    with txt_path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                det = parse_yolo_line(line, src="wasb")
                detections.append(det)
            except Exception as e:
                print(f"[警告] 跳过非法 WASB 标签行: {txt_path}:{line_idx} ({e})")
    return detections


def read_merged_labels_for_stem(labels_dir: Path, stem: str) -> List[Detection]:
    txt_path = labels_dir / f"{stem}.txt"
    if not txt_path.exists():
        return []

    detections: List[Detection] = []
    with txt_path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                det = parse_yolo_line(line, src="merged")
                detections.append(det)
            except Exception as e:
                print(f"[警告] 跳过非法融合标签行: {txt_path}:{line_idx} ({e})")
    return detections


def yolo_result_to_pixel_dets(result, orig_w: int, orig_h: int, conf_thresh: float) -> List[Detection]:
    detections: List[Detection] = []
    if result.boxes is None:
        return detections

    boxes = result.boxes.xywhn
    classes = result.boxes.cls
    confidences = result.boxes.conf

    for box, cls, conf in zip(boxes, classes, confidences):
        cf = float(conf)
        if cf < conf_thresh:
            continue
        x_n, y_n, w_n, h_n = [float(v) for v in box.tolist()]
        x_px = x_n * orig_w
        y_px = y_n * orig_h
        w_px = w_n * orig_w
        h_px = h_n * orig_h
        detections.append({
            "cls": int(float(cls)),
            "x": x_px,
            "y": y_px,
            "w": w_px,
            "h": h_px,
            "conf": cf,
            "src": "yolo",
        })
    return detections


def pixel_to_norm(det: Detection, orig_w: int, orig_h: int) -> Detection:
    x = min(max(float(det["x"]) / float(orig_w), 0.0), 1.0)
    y = min(max(float(det["y"]) / float(orig_h), 0.0), 1.0)
    w = min(max(float(det["w"]) / float(orig_w), 0.0), 1.0)
    h = min(max(float(det["h"]) / float(orig_h), 0.0), 1.0)
    out = dict(det)
    out["x"] = x
    out["y"] = y
    out["w"] = w
    out["h"] = h
    return out


def write_yolo_file(path: Path, dets: List[Detection], save_empty: bool):
    if not dets and not save_empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if not dets:
        path.write_text("", encoding="utf-8")
        return
    lines = [f"{int(d['cls'])} {d['x']:.6f} {d['y']:.6f} {d['w']:.6f} {d['h']:.6f}" for d in dets]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve_visual_outputs(args, output_folder: Path) -> Tuple[Path, Path]:
    vis_dir = Path(args.visualize_dir) if args.visualize_dir else output_folder.parent / f"{output_folder.name}_visualized"
    video_path = Path(args.video_path) if args.video_path else output_folder.parent / f"{output_folder.name}_visualized.mp4"
    return vis_dir, video_path


def _draw_one_frame(image, detections: List[Detection]):
    import cv2

    h_img, w_img = image.shape[:2]
    palette = [
        (0, 255, 0),
        (0, 128, 255),
        (255, 0, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 255, 0),
    ]

    for det in detections:
        cls_id = int(det["cls"])
        x = float(det["x"])
        y = float(det["y"])
        w = float(det["w"])
        h = float(det["h"])

        x1 = int(round((x - w / 2.0) * w_img))
        y1 = int(round((y - h / 2.0) * h_img))
        x2 = int(round((x + w / 2.0) * w_img))
        y2 = int(round((y + h / 2.0) * h_img))

        x1 = max(0, min(w_img - 1, x1))
        y1 = max(0, min(h_img - 1, y1))
        x2 = max(0, min(w_img - 1, x2))
        y2 = max(0, min(h_img - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        color = palette[cls_id % len(palette)]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"cls:{cls_id}", (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    return image


def run_visualization_from_labels(images: List[Path], labels_dir: Path, args) -> Dict[str, str]:
    try:
        import cv2
    except Exception as e:
        raise RuntimeError(f"启用可视化需要安装 opencv-python，当前导入失败: {e}")

    vis_dir, video_path = _resolve_visual_outputs(args, labels_dir)
    save_images = bool(args.visualize)
    save_video = bool(args.visualize_video)

    if save_images:
        vis_dir.mkdir(parents=True, exist_ok=True)
    if save_video:
        video_path.parent.mkdir(parents=True, exist_ok=True)

    video_writer = None
    video_size = None
    frames_written = 0
    total_boxes_drawn = 0

    for idx, img_path in enumerate(images, start=1):
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"[警告] 可视化读取图片失败，跳过: {img_path}")
            continue

        stem = img_path.stem
        detections = read_merged_labels_for_stem(labels_dir, stem)
        total_boxes_drawn += len(detections)

        annotated = _draw_one_frame(frame, detections)

        if save_images:
            out_img = vis_dir / img_path.name
            cv2.imwrite(str(out_img), annotated)

        if save_video:
            h, w = annotated.shape[:2]
            if video_writer is None:
                video_size = (w, h)
                fourcc = cv2.VideoWriter_fourcc(*args.video_fourcc)
                video_writer = cv2.VideoWriter(str(video_path), fourcc, float(args.video_fps), video_size)
                if not video_writer.isOpened():
                    raise RuntimeError(f"视频写入器创建失败: {video_path}")

            frame_to_write = annotated
            if video_size and (w, h) != video_size:
                frame_to_write = cv2.resize(annotated, video_size)
            video_writer.write(frame_to_write)

        frames_written += 1

        if idx % 200 == 0:
            print(f"可视化已处理 {idx}/{len(images)} 帧...")

    if video_writer is not None:
        video_writer.release()

    return {
        "frames_written": str(frames_written),
        "boxes_drawn": str(total_boxes_drawn),
        "visualize_dir": str(vis_dir) if save_images else "",
        "video_path": str(video_path) if save_video else "",
    }


def run_hybrid_predict(args):
    validate_args(args)

    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)
    wasb_labels_dir = Path(args.wasb_labels_dir)

    if not input_folder.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_folder}")
    if not wasb_labels_dir.exists():
        raise FileNotFoundError(f"WASB 标签目录不存在: {wasb_labels_dir}")

    output_folder.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.yolo_model)

    images = sorted([p for p in input_folder.iterdir() if p.is_file() and is_image_file(p)])
    if not images:
        print(f"[提示] 输入目录中未找到图片: {input_folder}")
        return
    if args.max_images > 0:
        images = images[: args.max_images]

    total = 0
    total_yolo_all = 0
    total_yolo_kept = 0
    total_yolo_dropped_roi = 0
    total_wasb = 0
    total_merged = 0
    missing_wasb_files = 0

    for img_path in images:
        total += 1
        stem = img_path.stem

        try:
            results = model(str(img_path), verbose=False)
            result = results[0]
        except Exception as e:
            print(f"[错误] YOLO 推理失败: {img_path.name} ({e})")
            continue

        yolo_pixel_dets = yolo_result_to_pixel_dets(result, args.orig_w, args.orig_h, args.conf)
        total_yolo_all += len(yolo_pixel_dets)

        yolo_kept_pixel: List[Detection] = []
        for det in yolo_pixel_dets:
            if inside_roi(det, args.left, args.top, args.right, args.bottom, args.inside_policy):
                total_yolo_dropped_roi += 1
            else:
                yolo_kept_pixel.append(det)
                total_yolo_kept += 1

        yolo_kept_norm = [pixel_to_norm(d, args.orig_w, args.orig_h) for d in yolo_kept_pixel]

        wasb_norm = read_wasb_labels_for_stem(wasb_labels_dir, stem)
        if not (wasb_labels_dir / f"{stem}.txt").exists():
            missing_wasb_files += 1
        total_wasb += len(wasb_norm)

        merged = yolo_kept_norm + wasb_norm
        merged = nms_by_class(merged, args.nms_iou)
        total_merged += len(merged)

        out_path = output_folder / f"{stem}.txt"
        write_yolo_file(out_path, merged, save_empty=not args.no_save_empty)

        if total % 200 == 0:
            print(f"已处理 {total}/{len(images)} 帧...")

    vis_stats = None
    if args.visualize or args.visualize_video:
        print("\n开始执行融合结果可视化...")
        vis_stats = run_visualization_from_labels(images=images, labels_dir=output_folder, args=args)

    print("\n" + "=" * 68)
    print("融合推理完成")
    print("=" * 68)
    print(f"总帧数                     : {total}")
    print(f"YOLO 原始框总数            : {total_yolo_all}")
    print(f"YOLO 保留（ROI 外）框总数  : {total_yolo_kept}")
    print(f"YOLO 丢弃（ROI 内）框总数  : {total_yolo_dropped_roi}")
    print(f"WASB+FP 框总数             : {total_wasb}")
    print(f"融合后框总数               : {total_merged}")
    print(f"缺失 WASB 标签文件帧数      : {missing_wasb_files}")
    if vis_stats is not None:
        print(f"可视化写入帧数              : {vis_stats['frames_written']}")
        print(f"可视化绘制框总数            : {vis_stats['boxes_drawn']}")
        if vis_stats["visualize_dir"]:
            print(f"可视化图片目录              : {vis_stats['visualize_dir']}")
        if vis_stats["video_path"]:
            print(f"可视化视频路径              : {vis_stats['video_path']}")
    print(f"输出目录                   : {output_folder}")
    print("=" * 68)


if __name__ == "__main__":
    run_hybrid_predict(parse_args())
