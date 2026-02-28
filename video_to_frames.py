import cv2
import os
import sys
import argparse
from pathlib import Path


def extract_frames_from_video(video_path, output_folder, frame_interval=1):
    """
    将视频分解为帧图像
    
    Args:
        video_path (str): 输入视频文件路径
        output_folder (str): 输出文件夹路径
        frame_interval (int): 提取帧的间隔，默认为1（每一帧都提取）
    """
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 {video_path}")
        return False
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"视频信息:")
    print(f"  FPS: {fps}")
    print(f"  总帧数: {total_frames}")
    print(f"  时长: {duration:.2f}秒")
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 按照指定间隔提取帧
        if frame_count % frame_interval == 0:
            # 使用时间戳作为文件名，确保唯一性和可排序性
            timestamp_ms = int(frame_count / fps * 1000)
            output_path = os.path.join(output_folder, f"{timestamp_ms:08d}.jpg")
            
            # 保存帧为JPG格式
            success = cv2.imwrite(output_path, frame)
            if success:
                saved_count += 1
                print(f"已保存帧: {output_path}")
            else:
                print(f"保存失败: {output_path}")
        
        frame_count += 1
    
    cap.release()
    
    print(f"\n完成！共处理 {frame_count} 帧，保存了 {saved_count} 张图片到 {output_folder}")
    return True


def main():
    parser = argparse.ArgumentParser(description="将MP4视频转换为YOLO预测所需的图像帧")
    parser.add_argument("--video", type=str, required=True, help="输入的MP4视频文件路径")
    parser.add_argument("--output", type=str, required=True, help="输出文件夹路径")
    parser.add_argument("--interval", type=int, default=1, help="帧提取间隔，默认为1（每一帧都提取）")
    
    args = parser.parse_args()
    
    # 检查视频文件是否存在
    if not os.path.exists(args.video):
        print(f"错误：视频文件 {args.video} 不存在")
        sys.exit(1)
    
    # 检查视频文件扩展名
    if not args.video.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm')):
        print(f"警告：文件可能不是视频格式，但仍将尝试处理")
    
    # 提取帧
    success = extract_frames_from_video(args.video, args.output, args.interval)
    
    if success:
        print(f"\n视频帧提取完成！图像已保存到: {args.output}")
        print(f"现在你可以使用 predict.py 对这些图像进行YOLO预测")
        print(f"例如: python predict.py --input_folder {args.output} --output_folder ./output_labels")
    else:
        print("\n视频帧提取失败！")
        sys.exit(1)


if __name__ == "__main__":
    main()
    
'''
使用方法
python video_to_frames.py --video left.mp4 --output ./left_frames --interval 1      
'''