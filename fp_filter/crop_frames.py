"""
将图片按指定矩形区域裁剪
原图区域尺寸为：1920*1080，裁剪后区域尺寸为：586*288
裁剪区域: 左上(left,top) 右下(right,bottom)，可通过参数指定
"""

import argparse
from pathlib import Path

from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="批量裁剪图片指定区域")
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="输入图片所在文件夹路径",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="裁剪后图片保存文件夹路径",
    )
    parser.add_argument(
        "--left",
        type=int,
        default=650,
        help="裁剪区域左边界 (默认: 650)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=51,
        help="裁剪区域上边界 (默认: 61)",
    )
    parser.add_argument(
        "--right",
        type=int,
        default=1236,
        help="裁剪区域右边界 (默认: 1236)",
    )
    parser.add_argument(
        "--bottom",
        type=int,
        default=339,
        help="裁剪区域下边界 (默认: 349)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    source_dir = Path(args.input)
    output_dir = Path(args.output)
    crop_box = (args.left, args.top, args.right, args.bottom)

    if not source_dir.exists():
        print(f"错误: 输入目录不存在: {source_dir}")
        return

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取所有 jpg 图片
    image_files = sorted(source_dir.glob("*.jpg"))
    total = len(image_files)

    if total == 0:
        print(f"在 {source_dir} 中未找到 .jpg 图片")
        return

    print(f"找到 {total} 张图片，开始裁剪...")
    success_count = 0
    error_count = 0

    for i, img_path in enumerate(image_files):
        try:
            with Image.open(img_path) as img:
                # 裁剪指定区域
                cropped = img.crop(crop_box)
                # 保存到输出目录，保持原文件名
                out_path = output_dir / img_path.name
                cropped.save(out_path, quality=95)
            success_count += 1

            # 每处理 5000 张打印进度
            if (i + 1) % 500 == 0:
                print(f"  已处理 {i + 1}/{total} 张...")
        except Exception as e:
            error_count += 1
            print(f"  处理失败: {img_path.name} - {e}")

    print(f"\n完成! 成功: {success_count}, 失败: {error_count}")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()

# python crop_yuedong_frames.py -i right_yuedong_frames -o right_yuedong_up

# python crop_frames.py --input ..\clip1_yolo --output ..\datasets\tennis_predict\match1\clip1
