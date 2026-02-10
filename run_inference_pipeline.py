"""
完整推理 Pipeline：WASB 球检测 -> FP 过滤 -> 结果可视化

此脚本自动串联三个步骤：
1. 运行 WASB 模型对新数据集进行初步检测（src/main.py）
2. 使用 FP 过滤器剔除误检（fp_filter/inference.py）
3. 生成对比可视化视频（fp_filter/visualize_filtered.py）

使用示例：
    python run_inference_pipeline.py
    
可选参数：
    python run_inference_pipeline.py ^
        --dataset-root "datasets/tennis_predict" ^
        --wasb-weight "pretrained_weights/wasb_tennis_best.pth.tar" ^
        --fp-model "fp_filter/patch_outputs/model_resnet/best.pth" ^
        --output-base "pipeline_outputs" ^
        --fps 25 ^
        --step 3

"""

import os
import os.path as osp
import sys
import argparse
import subprocess
import re
import glob
import logging
from datetime import datetime
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)


class InferencePipeline:
    """完整推理流程的封装类"""
    
    def __init__(self, args):
        """
        初始化 Pipeline
        
        Args:
            args: 命令行参数对象
        """
        self.args = args
        
        # 项目根目录（脚本所在目录）
        self.root_dir = Path(__file__).parent.absolute()
        
        # 关键路径
        self.dataset_root = self.root_dir / args.dataset_root
        self.wasb_weight = self.root_dir / args.wasb_weight
        self.fp_model = self.root_dir / args.fp_model
        
        # 输出目录（带时间戳）
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_dir = self.root_dir / args.output_base / timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 阶段性输出目录
        self.stage1_output = self.output_dir / "stage1_wasb_detection"
        self.stage2_output = self.output_dir / "stage2_fp_filtered"
        self.stage3_output = self.output_dir / "stage3_visualizations"
        
        log.info(f"Pipeline 输出目录: {self.output_dir}")
        
        # 验证必要文件存在
        self._validate_prerequisites()
    
    def _validate_prerequisites(self):
        """验证必要的文件和目录是否存在"""
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"数据集目录不存在: {self.dataset_root}")
        
        if not self.wasb_weight.exists():
            raise FileNotFoundError(f"WASB 权重文件不存在: {self.wasb_weight}")
        
        if not self.fp_model.exists():
            raise FileNotFoundError(f"FP 过滤器模型不存在: {self.fp_model}")
        
        log.info("✓ 所有必要文件验证通过")
    
    def run(self):
        """执行完整的推理 Pipeline"""
        log.info("="*80)
        log.info("开始执行完整推理 Pipeline")
        log.info("="*80)
        
        try:
            # Stage 1: WASB 检测
            stage1_success = self._run_stage1_wasb_detection()
            if not stage1_success:
                log.error("Stage 1 失败，终止 Pipeline")
                return False
            
            # Stage 2: FP 过滤
            stage2_success = self._run_stage2_fp_filtering()
            if not stage2_success:
                log.error("Stage 2 失败，终止 Pipeline")
                return False
            
            # Stage 3: 可视化
            stage3_success = self._run_stage3_visualization()
            if not stage3_success:
                log.error("Stage 3 失败")
                return False
            
            log.info("="*80)
            log.info("✓ Pipeline 执行成功完成！")
            log.info(f"所有结果保存在: {self.output_dir}")
            log.info("="*80)
            
            # 打印结果摘要
            self._print_summary()
            
            return True
            
        except Exception as e:
            log.error(f"Pipeline 执行出错: {e}", exc_info=True)
            return False
    
    def _run_stage1_wasb_detection(self):
        """
        Stage 1: 运行 WASB 模型进行初步球检测
        
        Returns:
            bool: 是否成功
        """
        log.info("\n" + "="*80)
        log.info("Stage 1: WASB 球检测")
        log.info("="*80)
        
        self.stage1_output.mkdir(parents=True, exist_ok=True)
        
        # 构建命令
        src_dir = self.root_dir / "src"
        
        # Hydra 配置：指定输出目录
        hydra_output = self.stage1_output.as_posix()
        
        cmd = [
            sys.executable,  # 使用当前 Python 解释器
            "main.py",
            "--config-name=eval",
            f"dataset=tennis_predict",
            f"model=wasb",
            f"detector.model_path={self.wasb_weight.as_posix()}",
            f"runner.split=test",
            f"runner.vis_result=True",
            f"detector.step={self.args.step}",
            f"hydra.run.dir={hydra_output}",  # 指定输出目录
        ]
        
        log.info(f"执行命令: {' '.join(cmd)}")
        log.info(f"工作目录: {src_dir}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(src_dir),
                check=True,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='ignore'
            )
            
            log.info("Stage 1 输出:")
            for line in result.stdout.split('\n'):
                if line.strip():
                    log.info(f"  {line}")
            
            # 查找生成的 CSV 文件
            csv_files = list(self.stage1_output.glob("*_predictions.csv"))
            if not csv_files:
                log.error("未找到生成的 predictions.csv 文件")
                return False
            
            log.info(f"✓ Stage 1 完成，生成了 {len(csv_files)} 个检测结果文件")
            for csv_file in csv_files:
                log.info(f"  - {csv_file.name}")
            
            return True
            
        except subprocess.CalledProcessError as e:
            log.error(f"Stage 1 执行失败: {e}")
            log.error(f"错误输出: {e.stderr}")
            return False
    
    def _run_stage2_fp_filtering(self):
        """
        Stage 2: 使用 FP 过滤器剔除误检
        
        Returns:
            bool: 是否成功
        """
        log.info("\n" + "="*80)
        log.info("Stage 2: FP 误检过滤")
        log.info("="*80)
        
        self.stage2_output.mkdir(parents=True, exist_ok=True)
        
        # 查找所有待处理的 CSV 文件
        csv_files = list(self.stage1_output.glob("*_predictions.csv"))
        
        if not csv_files:
            log.error("Stage 1 未生成任何 CSV 文件")
            return False
        
        success_count = 0
        fp_filter_dir = self.root_dir / "fp_filter"
        
        for csv_file in csv_files:
            log.info(f"\n处理: {csv_file.name}")
            
            # 生成输出文件名
            output_csv = self.stage2_output / csv_file.name.replace("_predictions.csv", "_predictions_filtered.csv")
            
            # 构建命令
            cmd = [
                sys.executable,
                "inference.py",
                "--csv", csv_file.as_posix(),
                "--dataset-root", self.dataset_root.as_posix(),
                "--model", self.fp_model.as_posix(),
                "--output", output_csv.as_posix(),
                "--threshold", str(self.args.threshold),
            ]
            
            log.info(f"执行命令: {' '.join(cmd)}")
            
            try:
                result = subprocess.run(
                    cmd,
                    cwd=str(fp_filter_dir),
                    check=True,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='ignore'
                )
                
                # 打印关键输出信息
                for line in result.stdout.split('\n'):
                    if any(keyword in line for keyword in ['过滤', '保留', '移除', 'Filtered', 'Removed']):
                        log.info(f"  {line}")
                
                log.info(f"✓ 已生成: {output_csv.name}")
                success_count += 1
                
            except subprocess.CalledProcessError as e:
                log.error(f"处理 {csv_file.name} 失败: {e}")
                log.error(f"错误输出: {e.stderr}")
        
        if success_count == 0:
            return False
        
        log.info(f"\n✓ Stage 2 完成，成功处理 {success_count}/{len(csv_files)} 个文件")
        return True
    
    def _run_stage3_visualization(self):
        """
        Stage 3: 生成对比可视化视频
        
        Returns:
            bool: 是否成功
        """
        log.info("\n" + "="*80)
        log.info("Stage 3: 结果可视化")
        log.info("="*80)
        
        self.stage3_output.mkdir(parents=True, exist_ok=True)
        
        # 查找过滤后的 CSV 文件
        filtered_csv_files = list(self.stage2_output.glob("*_filtered.csv"))
        
        if not filtered_csv_files:
            log.error("Stage 2 未生成任何过滤后的 CSV 文件")
            return False
        
        success_count = 0
        fp_filter_dir = self.root_dir / "fp_filter"
        
        for filtered_csv in filtered_csv_files:
            log.info(f"\n可视化: {filtered_csv.name}")
            
            # 找到对应的原始 CSV
            original_name = filtered_csv.name.replace("_filtered.csv", ".csv")
            original_csv = self.stage1_output / original_name
            
            if not original_csv.exists():
                log.warning(f"找不到原始 CSV: {original_csv.name}，跳过")
                continue
            
            # 生成视频文件名
            video_name = filtered_csv.stem.replace("_predictions_filtered", "_final_result.mp4")
            output_video = self.stage3_output / video_name
            
            # 构建命令
            cmd = [
                sys.executable,
                "visualize_filtered.py",
                "--csv", original_csv.as_posix(),
                "--filtered-csv", filtered_csv.as_posix(),
                "--dataset-root", self.stage1_output.as_posix(),  # 注意：这里使用 Stage 1 的输出作为图像源
                "--output-video", output_video.as_posix(),
                "--fps", str(self.args.fps),
            ]
            
            log.info(f"执行命令: {' '.join(cmd)}")
            
            try:
                result = subprocess.run(
                    cmd,
                    cwd=str(fp_filter_dir),
                    check=True,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='ignore'
                )
                
                # 打印进度信息
                for line in result.stdout.split('\n'):
                    if any(keyword in line for keyword in ['保存', 'Saved', '生成', 'Generated', '完成']):
                        log.info(f"  {line}")
                
                if output_video.exists():
                    log.info(f"✓ 视频已生成: {output_video.name}")
                    success_count += 1
                else:
                    log.warning(f"视频文件未生成: {output_video.name}")
                
            except subprocess.CalledProcessError as e:
                log.error(f"可视化 {filtered_csv.name} 失败: {e}")
                log.error(f"错误输出: {e.stderr}")
        
        if success_count == 0:
            return False
        
        log.info(f"\n✓ Stage 3 完成，成功生成 {success_count}/{len(filtered_csv_files)} 个视频")
        return True
    
    def _print_summary(self):
        """打印结果摘要"""
        log.info("\n" + "="*80)
        log.info("结果摘要")
        log.info("="*80)
        
        # Stage 1 结果
        csv_files = list(self.stage1_output.glob("*_predictions.csv"))
        log.info(f"\n📊 Stage 1 - WASB 检测结果:")
        log.info(f"  位置: {self.stage1_output}")
        log.info(f"  生成文件数: {len(csv_files)}")
        
        # Stage 2 结果
        filtered_files = list(self.stage2_output.glob("*_filtered.csv"))
        log.info(f"\n🔍 Stage 2 - FP 过滤结果:")
        log.info(f"  位置: {self.stage2_output}")
        log.info(f"  生成文件数: {len(filtered_files)}")
        
        # Stage 3 结果
        video_files = list(self.stage3_output.glob("*.mp4"))
        log.info(f"\n🎬 Stage 3 - 可视化视频:")
        log.info(f"  位置: {self.stage3_output}")
        log.info(f"  生成视频数: {len(video_files)}")
        for video in video_files:
            size_mb = video.stat().st_size / (1024 * 1024)
            log.info(f"  - {video.name} ({size_mb:.1f} MB)")
        
        log.info(f"\n📁 完整输出目录: {self.output_dir}")
        log.info("="*80)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='完整推理 Pipeline：WASB 检测 -> FP 过滤 -> 可视化',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 输入路径
    parser.add_argument(
        '--dataset-root',
        type=str,
        default='datasets/tennis_predict',
        help='原始数据集根目录（相对于项目根目录）'
    )
    
    parser.add_argument(
        '--wasb-weight',
        type=str,
        default='pretrained_weights/wasb_tennis_best.pth.tar',
        help='WASB 模型权重文件路径'
    )
    
    parser.add_argument(
        '--fp-model',
        type=str,
        default='fp_filter/patch_outputs/model_resnet/best.pth',
        help='FP 过滤器模型权重文件路径'
    )
    
    # 输出路径
    parser.add_argument(
        '--output-base',
        type=str,
        default='pipeline_outputs',
        help='Pipeline 输出基础目录'
    )
    
    # 推理参数
    parser.add_argument(
        '--step',
        type=int,
        default=3,
        help='WASB 检测步长（1=逐帧检测，3=每3帧检测一次）'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='FP 过滤器阈值（0-1，越高越严格）'
    )
    
    parser.add_argument(
        '--fps',
        type=int,
        default=25,
        help='输出视频帧率'
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 创建并运行 Pipeline
    pipeline = InferencePipeline(args)
    success = pipeline.run()
    
    # 退出码
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
