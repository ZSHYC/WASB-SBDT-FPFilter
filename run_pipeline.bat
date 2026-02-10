@echo off
REM =====================================================================
REM 完整推理 Pipeline 启动脚本 (Windows)
REM 
REM 功能：一键完成 WASB 检测 -> FP 过滤 -> 可视化全流程
REM 使用：双击运行或在命令行执行 run_pipeline.bat
REM =====================================================================

echo.
echo ========================================================================
echo                   网球检测完整推理 Pipeline
echo ========================================================================
echo.
echo 即将开始执行三个阶段：
echo   [1] WASB 球检测
echo   [2] FP 误检过滤
echo   [3] 结果可视化
echo.
echo 请确保：
echo   - 数据已放在 datasets/tennis_predict/ 目录下
echo   - 模型权重文件已就位
echo.
echo ========================================================================
echo.

REM 等待用户确认
pause

REM 激活虚拟环境（如果使用）
REM call venv\Scripts\activate.bat

REM 运行 Pipeline（使用默认参数）
python run_inference_pipeline.py

REM 如果需要自定义参数，可以修改下面的命令
REM python run_inference_pipeline.py --step 3 --threshold 0.5 --fps 25

echo.
echo ========================================================================
echo Pipeline 执行完毕！
echo.
echo 结果保存在 pipeline_outputs 目录下
echo ========================================================================
echo.

pause
