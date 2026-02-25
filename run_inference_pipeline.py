"""
完整推理 Pipeline：WASB 球检测 -> FP 过滤 -> 结果可视化 -> YOLO 标签生成 -> 原图尺度标签生成

此脚本自动串联五个步骤：
1. 运行 WASB 模型对新数据集进行初步检测（src/main.py）
2. 使用 FP 过滤器剔除误检（fp_filter/inference.py）
3. 生成对比可视化视频（fp_filter/visualize_filtered.py）
4. 将过滤后的 CSV 转换为逐帧 YOLO txt 标签（fp_filter/csv_to_yolo_txt.py）
5. 将过滤后的 CSV 转换为原图尺度 YOLO txt 标签（fp_filter/csv_to_original_yolo.py）

使用示例：
    python run_inference_pipeline.py --fp-model "fp_filter/patch_outputs/model_resnet_v3/best.pth" 
    
可选参数：
    python run_inference_pipeline.py ^
        --dataset-root "datasets/tennis_predict" ^
        --wasb-weight "pretrained_weights/wasb_tennis_best.pth.tar" ^
        --fp-model "fp_filter/patch_outputs/model_resnet/best.pth" ^
        --output-base "pipeline_outputs" ^
        --fps 25 ^
        --step 1 ^
        --box-size 15 ^
        --class-id 0 ^
        --crop-left 650 --crop-top 51 ^
        --orig-w 1920 --orig-h 1080

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
        # 保存传入的命令行参数对象，后续方法读取配置
        self.args = args

        # 项目根目录：使用脚本文件的父目录作为项目根（跨平台可靠）
        # Path(__file__).parent.absolute() 返回脚本所在目录的绝对路径
        self.root_dir = Path(__file__).parent.absolute()

        # 将命令行传入的相对路径转换为以项目根为基准的 Path 对象
        # 例如 args.dataset_root 默认是 'datasets/tennis_predict'
        self.dataset_root = self.root_dir / args.dataset_root
        self.wasb_weight = self.root_dir / args.wasb_weight
        self.fp_model = self.root_dir / args.fp_model

        # 生成输出目录，使用时间戳保证每次运行的目录唯一，便于追溯
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # output_base 默认 'pipeline_outputs'，最终路径为 <root>/pipeline_outputs/<timestamp>
        self.output_dir = self.root_dir / args.output_base / timestamp
        # 创建目录（存在也不报错）
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 为五个阶段分别创建子目录路径变量（实际创建通常在各阶段开始时）
        self.stage1_output = self.output_dir / "stage1_wasb_detection"
        self.stage2_output = self.output_dir / "stage2_fp_filtered"
        self.stage3_output = self.output_dir / "stage3_visualizations"
        self.stage4_output = self.output_dir / "stage4_yolo_labels"
        self.stage5_output = self.output_dir / "stage5_original_yolo_labels"

        # 打印输出目录信息，方便用户查看
        log.info(f"Pipeline 输出目录: {self.output_dir}")

        # 在继续执行之前检查必要的文件和目录是否存在（如数据集、模型权重）
        self._validate_prerequisites()
    
    def _validate_prerequisites(self):
        """验证必要的文件和目录是否存在"""
        # 检查数据集目录是否存在，否则直接抛出错误并中断
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"数据集目录不存在: {self.dataset_root}")

        # 检查 WASB 模型权重文件是否存在
        if not self.wasb_weight.exists():
            raise FileNotFoundError(f"WASB 权重文件不存在: {self.wasb_weight}")

        # 检查 FP 过滤器模型权重是否存在
        if not self.fp_model.exists():
            raise FileNotFoundError(f"FP 过滤器模型不存在: {self.fp_model}")

        # 所有前置条件满足，记录日志
        log.info("✓ 所有必要文件验证通过")
    
    def run(self):
        """执行完整的推理 Pipeline"""
        log.info("="*80)
        log.info("开始执行完整推理 Pipeline")
        log.info("="*80)
        
        try:
            # 依次执行三个阶段的方法。
            # 每个阶段返回 True/False 表示是否成功，失败则提前终止 Pipeline
            stage1_success = self._run_stage1_wasb_detection()
            if not stage1_success:
                # 如果 Stage 1 失败（例如未生成 CSV），记录错误并停止
                log.error("Stage 1 失败，终止 Pipeline")
                return False

            stage2_success = self._run_stage2_fp_filtering()
            if not stage2_success:
                log.error("Stage 2 失败，终止 Pipeline")
                return False

            stage3_success = self._run_stage3_visualization()
            if not stage3_success:
                log.error("Stage 3 失败")
                return False

            stage4_success = self._run_stage4_yolo_labels()
            if not stage4_success:
                log.error("Stage 4 失败")
                return False

            stage5_success = self._run_stage5_original_yolo_labels()
            if not stage5_success:
                log.error("Stage 5 失败")
                return False

            # 所有阶段成功完成，打印成功信息
            log.info("="*80)
            log.info("✓ Pipeline 执行成功完成！")
            log.info(f"所有结果保存在: {self.output_dir}")
            log.info("="*80)

            # 打印摘要信息，便于用户查看结果统计
            self._print_summary()

            return True

        except Exception as e:
            # 捕获任意未处理的异常并记录堆栈信息，便于调试
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
        
        # 确保 Stage1 输出目录存在
        self.stage1_output.mkdir(parents=True, exist_ok=True)

        # src_dir 指向项目中的 src 目录，我们将在该目录下运行 src/main.py
        src_dir = self.root_dir / "src"

        # 将 Stage1 的输出目录传给 Hydra，使得 src/main.py 将输出写入该目录
        # 使用 as_posix() 生成统一的路径格式，避免反斜杠转义问题
        hydra_output = self.stage1_output.as_posix()

        # 构建要执行的子进程命令，使用当前 Python 解释器（sys.executable）
        # 这里通过向 main.py 传入 CLI 覆盖项来控制数据集、模型路径、步长等
        cmd = [
            sys.executable,            # 当前 Python 可执行文件路径
            "main.py",               # 要执行的脚本（位于 src/ 下）
            "--config-name=eval",   # 指定使用的 Hydra 配置
            f"dataset=tennis_predict",
            f"model=wasb",
            f"detector.model_path={self.wasb_weight.as_posix()}",
            f"runner.split=test",
            f"runner.vis_result=True",
            f"detector.step={self.args.step}",
            f"hydra.run.dir={hydra_output}",  # 覆盖 Hydra 输出目录
        ]

        # 记录将执行的命令与工作目录，便于排查
        log.info(f"执行命令: {' '.join(cmd)}")
        log.info(f"工作目录: {src_dir}")

        try:
            # 以子进程方式运行 src/main.py，capture_output=True 用于捕获 stdout/stderr
            result = subprocess.run(
                cmd,
                cwd=str(src_dir),          # 在 src 目录下执行命令
                check=True,                # 子进程返回非0时抛出 CalledProcessError
                capture_output=True,       # 捕获输出方便日志记录
                text=True,                 # 将输出解码为文本（不是 bytes）
                encoding='utf-8',
                errors='ignore'
            )

            # 将子进程的标准输出逐行记录到日志中（避免大量一次性输出）
            log.info("Stage 1 输出:")
            for line in result.stdout.split('\n'):
                if line.strip():
                    log.info(f"  {line}")

            # 如果有 stderr 输出也记录，便于诊断（某些脚本会把重要信息写到 stderr）
            if result.stderr and result.stderr.strip():
                log.warning("Stage 1 stderr:\n" + result.stderr.strip())

            # 扫描指定输出目录下的 *_predictions.csv 文件以确认生成结果
            csv_files = list(self.stage1_output.glob("*_predictions.csv"))
            if not csv_files:
                # 若没有找到 CSV，说明可能主脚本运行时出错或配置不匹配
                log.error("未找到生成的 predictions.csv 文件")
                return False

            # 列出生成的 CSV 文件，便于用户核对
            log.info(f"✓ Stage 1 完成，生成了 {len(csv_files)} 个检测结果文件")
            for csv_file in csv_files:
                log.info(f"  - {csv_file.name}")

            return True

        except subprocess.CalledProcessError as e:
            # 子进程执行失败时打印错误信息（包括 stderr）以便定位问题
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
        
        # 确保 Stage2 输出目录存在
        self.stage2_output.mkdir(parents=True, exist_ok=True)

        # 从 Stage1 目录查找所有需要处理的 predictions.csv
        csv_files = list(self.stage1_output.glob("*_predictions.csv"))

        if not csv_files:
            # 没有找到 CSV，说明 Stage1 可能没有成功
            log.error("Stage 1 未生成任何 CSV 文件")
            return False

        success_count = 0
        # FP 过滤脚本所在目录（我们将在该目录下执行 inference.py）
        fp_filter_dir = self.root_dir / "fp_filter"

        # 遍历每个 CSV 文件并调用 fp_filter/inference.py 进行二分类过滤
        for csv_file in csv_files:
            log.info(f"\n处理: {csv_file.name}")

            # 构建过滤后输出文件名，例如 xxx_predictions_filtered.csv
            output_csv = self.stage2_output / csv_file.name.replace("_predictions.csv", "_predictions_filtered.csv")

            # 构造子进程命令：在 fp_filter 目录下执行 inference.py
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
                # 调用子进程并捕获输出
                result = subprocess.run(
                    cmd,
                    cwd=str(fp_filter_dir),
                    check=True,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='ignore'
                )

                # 从子进程输出中筛选关键字（中文或英文）并记录到日志
                for line in result.stdout.split('\n'):
                    if any(keyword in line for keyword in ['过滤', '保留', '移除', 'Filtered', 'Removed']):
                        log.info(f"  {line}")

                # 如果子进程在 stderr 中有信息，也记录出来以便排查
                if result.stderr and result.stderr.strip():
                    log.warning("子进程 stderr:\n" + result.stderr.strip())

                # 确认 output_csv 文件已实际生成；只有存在时才计为成功
                if output_csv.exists():
                    log.info(f"✓ 已生成: {output_csv.name}")
                    success_count += 1
                else:
                    # 如果预期的输出文件不存在，则记录警告并继续（可能 inference.py 保存到其他位置）
                    log.warning(f"子进程执行成功但未找到输出文件: {output_csv.as_posix()}")

            except subprocess.CalledProcessError as e:
                # 若某个 CSV 的处理失败，记录错误并继续处理下一个文件
                log.error(f"处理 {csv_file.name} 失败: {e}")
                log.error(f"错误输出: {e.stderr}")

        if success_count == 0:
            # 所有文件均处理失败，返回 False
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
        
        # 确保 Stage3 输出目录存在
        self.stage3_output.mkdir(parents=True, exist_ok=True)

        # 查找所有在 Stage2 生成的过滤后 CSV
        filtered_csv_files = list(self.stage2_output.glob("*_filtered.csv"))

        if not filtered_csv_files:
            log.error("Stage 2 未生成任何过滤后的 CSV 文件")
            return False

        success_count = 0
        # visualize 脚本也在 fp_filter 目录下
        fp_filter_dir = self.root_dir / "fp_filter"

        for filtered_csv in filtered_csv_files:
            log.info(f"\n可视化: {filtered_csv.name}")

            # 根据过滤后的文件名推理出原始 CSV 名称（替换后缀）
            original_name = filtered_csv.name.replace("_filtered.csv", ".csv")
            original_csv = self.stage1_output / original_name

            # 如果找不到原始 CSV，就无法可视化，跳过该文件
            if not original_csv.exists():
                log.warning(f"找不到原始 CSV: {original_csv.name}，跳过")
                continue

            # 生成输出视频路径，文件名示例: match1_clip1_final_result.mp4
            video_name = filtered_csv.stem.replace("_predictions_filtered", "_final_result.mp4")
            output_video = self.stage3_output / video_name

            # 构建调用 visualize_filtered.py 的命令
            # 注意：--dataset-root 传入的是 Stage1 的输出目录（其中包含可视化帧）
            cmd = [
                sys.executable,
                "visualize_filtered.py",
                "--csv", original_csv.as_posix(),
                "--filtered-csv", filtered_csv.as_posix(),
                "--dataset-root", self.stage1_output.as_posix(),
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

                # 过滤输出中常见的表示“保存/生成”关键字用于记录
                for line in result.stdout.split('\n'):
                    if any(keyword in line for keyword in ['保存', 'Saved', '生成', 'Generated', '完成']):
                        log.info(f"  {line}")

                # 如果子进程在 stderr 中有信息，也记录出来以便排查
                if result.stderr and result.stderr.strip():
                    log.warning("子进程 stderr:\n" + result.stderr.strip())

                # 子进程返回后检查文件是否实际生成
                if output_video.exists():
                    log.info(f"✓ 视频已生成: {output_video.name}")
                    success_count += 1
                else:
                    # 如果脚本看似成功但文件不存在，记录警告以便调查
                    log.warning(f"视频文件未生成: {output_video.name}")

            except subprocess.CalledProcessError as e:
                # 某个可视化任务失败时记录错误并继续处理其他文件
                log.error(f"可视化 {filtered_csv.name} 失败: {e}")
                log.error(f"错误输出: {e.stderr}")

        if success_count == 0:
            return False

        log.info(f"\n✓ Stage 3 完成，成功生成 {success_count}/{len(filtered_csv_files)} 个视频")
        return True

    def _run_stage4_yolo_labels(self):
        """
        Stage 4: 将过滤后的 CSV 转换为逐帧 YOLO txt 标签

        依赖 fp_filter/csv_to_yolo_txt.py，对 Stage2 输出的每个
        *_predictions_filtered.csv 分别生成一个 <stem>_yolo_labels/ 子目录，
        其中每帧对应一个 YOLO 格式 .txt 文件。

        Returns:
            bool: 是否成功
        """
        log.info("\n" + "="*80)
        log.info("Stage 4: YOLO 标签生成")
        log.info("="*80)

        # 确保 Stage4 根输出目录存在
        self.stage4_output.mkdir(parents=True, exist_ok=True)

        # 从 Stage2 目录查找所有过滤后的 CSV
        filtered_csv_files = list(self.stage2_output.glob("*_filtered.csv"))
        if not filtered_csv_files:
            log.error("Stage 2 未生成任何过滤后的 CSV 文件，无法生成 YOLO 标签")
            return False

        fp_filter_dir = self.root_dir / "fp_filter"
        success_count = 0

        for filtered_csv in filtered_csv_files:
            log.info(f"\n处理: {filtered_csv.name}")

            # 每个 CSV 生成独立的子目录，例如 stage4_yolo_labels/match1_clip1_predictions_filtered_yolo_labels
            stem = filtered_csv.stem  # e.g. match1_clip1_predictions_filtered
            label_dir = self.stage4_output / f"{stem}_yolo_labels"

            cmd = [
                sys.executable,
                "csv_to_yolo_txt.py",
                "--csv", filtered_csv.as_posix(),
                "--image-root", self.dataset_root.as_posix(),
                "--output-dir", label_dir.as_posix(),
                "--box-size", str(self.args.box_size),
                "--class-id", str(self.args.class_id),
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

                # 将脚本输出的统计信息记录到日志
                for line in result.stdout.split('\n'):
                    if line.strip():
                        log.info(f"  {line}")

                if result.stderr and result.stderr.strip():
                    log.warning("子进程 stderr:\n" + result.stderr.strip())

                # 验证输出目录是否已创建且含有 txt 文件
                txt_files = list(label_dir.glob("*.txt")) if label_dir.exists() else []
                if txt_files:
                    log.info(f"✓ 已生成 YOLO 标签: {len(txt_files)} 个 txt -> {label_dir.name}")
                    success_count += 1
                else:
                    log.warning(f"未在 {label_dir} 中找到任何 .txt 文件")

            except subprocess.CalledProcessError as e:
                log.error(f"处理 {filtered_csv.name} 失败: {e}")
                log.error(f"错误输出: {e.stderr}")

        if success_count == 0:
            return False

        log.info(f"\n✓ Stage 4 完成，成功处理 {success_count}/{len(filtered_csv_files)} 个文件")
        return True

    def _run_stage5_original_yolo_labels(self):
        """
        Stage 5: 将过滤后的 CSV 转换为原图尺度 YOLO txt 标签

        依赖 fp_filter/csv_to_original_yolo.py，对 Stage2 输出的每个
        *_predictions_filtered.csv 分别生成一个 <stem>_orig_yolo_labels/ 子目录，
        其中每帧对应一个基于原图尺寸归一化的 YOLO 格式 .txt 文件。

        Returns:
            bool: 是否成功
        """
        log.info("\n" + "="*80)
        log.info("Stage 5: 原图尺度 YOLO 标签生成")
        log.info("="*80)

        self.stage5_output.mkdir(parents=True, exist_ok=True)

        filtered_csv_files = list(self.stage2_output.glob("*_filtered.csv"))
        if not filtered_csv_files:
            log.error("Stage 2 未生成任何过滤后的 CSV 文件，无法生成原图尺度 YOLO 标签")
            return False

        fp_filter_dir = self.root_dir / "fp_filter"
        success_count = 0

        for filtered_csv in filtered_csv_files:
            log.info(f"\n处理: {filtered_csv.name}")

            stem = filtered_csv.stem
            label_dir = self.stage5_output / f"{stem}_orig_yolo_labels"

            cmd = [
                sys.executable,
                "csv_to_original_yolo.py",
                "--csv", filtered_csv.as_posix(),
                "--image-root", self.dataset_root.as_posix(),
                "--output-dir", label_dir.as_posix(),
                "--crop-left", str(self.args.crop_left),
                "--crop-top", str(self.args.crop_top),
                "--orig-w", str(self.args.orig_w),
                "--orig-h", str(self.args.orig_h),
                "--box-size", str(self.args.box_size),
                "--class-id", str(self.args.class_id),
            ]

            if self.args.orig_no_save_empty:
                cmd.append("--no-save-empty")

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

                for line in result.stdout.split('\n'):
                    if line.strip():
                        log.info(f"  {line}")

                if result.stderr and result.stderr.strip():
                    log.warning("子进程 stderr:\n" + result.stderr.strip())

                txt_files = list(label_dir.glob("*.txt")) if label_dir.exists() else []
                if txt_files:
                    log.info(f"✓ 已生成原图尺度 YOLO 标签: {len(txt_files)} 个 txt -> {label_dir.name}")
                    success_count += 1
                else:
                    log.warning(f"未在 {label_dir} 中找到任何 .txt 文件")

            except subprocess.CalledProcessError as e:
                log.error(f"处理 {filtered_csv.name} 失败: {e}")
                log.error(f"错误输出: {e.stderr}")

        if success_count == 0:
            return False

        log.info(f"\n✓ Stage 5 完成，成功处理 {success_count}/{len(filtered_csv_files)} 个文件")
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
        
        # Stage 4 结果
        label_dirs = [d for d in self.stage4_output.iterdir() if d.is_dir()] if self.stage4_output.exists() else []
        total_txts = sum(len(list(d.glob("*.txt"))) for d in label_dirs)
        log.info(f"\n🏷️ Stage 4 - YOLO 标签生成:")
        log.info(f"  位置: {self.stage4_output}")
        log.info(f"  生成子目录数: {len(label_dirs)}")
        log.info(f"  总 txt 文件数: {total_txts}")

        # Stage 5 结果
        original_label_dirs = [d for d in self.stage5_output.iterdir() if d.is_dir()] if self.stage5_output.exists() else []
        total_original_txts = sum(len(list(d.glob("*.txt"))) for d in original_label_dirs)
        log.info(f"\n🎯 Stage 5 - 原图尺度 YOLO 标签生成:")
        log.info(f"  位置: {self.stage5_output}")
        log.info(f"  生成子目录数: {len(original_label_dirs)}")
        log.info(f"  总 txt 文件数: {total_original_txts}")

        log.info(f"\n📁 完整输出目录: {self.output_dir}")
        log.info("="*80)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='完整推理 Pipeline：WASB 检测 -> FP 过滤 -> 可视化 -> YOLO 标签 -> 原图尺度标签',
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
        default=1,
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

    # Stage 4 参数
    parser.add_argument(
        '--box-size',
        type=float,
        default=15.0,
        help='YOLO 标签中的固定框大小（像素），默认 15'
    )

    parser.add_argument(
        '--class-id',
        type=int,
        default=0,
        help='YOLO 标签中的类别 ID，默认 0'
    )

    # Stage 5 参数
    parser.add_argument(
        '--crop-left',
        type=int,
        default=650,
        help='原图尺度标签转换时的裁剪左偏移（像素）'
    )

    parser.add_argument(
        '--crop-top',
        type=int,
        default=51,
        help='原图尺度标签转换时的裁剪上偏移（像素）'
    )

    parser.add_argument(
        '--orig-w',
        type=int,
        default=1920,
        help='原图宽度（像素）'
    )

    parser.add_argument(
        '--orig-h',
        type=int,
        default=1080,
        help='原图高度（像素）'
    )

    parser.add_argument(
        '--orig-no-save-empty',
        action='store_true',
        default=False,
        help='Stage 5 不为无检测帧生成空 txt（默认生成）'
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
