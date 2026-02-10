"""
交互式标注工具：读取 manifest.csv，逐个显示 patch，通过键盘输入标签（1=球，0=非球），并实时保存。
极大提高手动标注效率，无需在 Excel 和图片浏览器之间切换。

使用示例（默认在fp_filter目录下执行）：cd fp_filter(别忘了！)
python label_patches.py --manifest patch_outputs/patches_train10_right/manifest.csv --size 512

从第100张开始标注：
python label_patches.py --manifest patch_outputs/patches_train6_right/manifest.csv --size 512 --start-from 100
"""

import os
import os.path as osp
import argparse
import pandas as pd
import cv2
import numpy as np

def label_patches(manifest_path, patch_size_display=256, start_from=0):
    """
    manifest_path: manifest.csv 的路径
    patch_size_display: 显示图片时的放大尺寸（原始 32x32 太小，建议放大查看）
    start_from: 从第几张图片开始标注（从0开始计数）
    """
    if not osp.isfile(manifest_path):
        print(f"错误：文件不存在 {manifest_path}")
        return

    # 读取 CSV
    print(f"正在读取: {manifest_path}")
    df = pd.read_csv(manifest_path)

    # 确保 label 列存在
    if "label" not in df.columns:
        df["label"] = "" # 初始化为空字符串或 NaN

    manifest_dir = osp.dirname(osp.abspath(manifest_path))
    
    # 统计进度
    total = len(df)
    
    # 全量索引列表（用于任意跳转），以及未标注掩码用于统计剩余
    full_indices = df.index.tolist()
    unlabeled_mask = df["label"].isna() | (df["label"].astype(str).str.strip() == "")
    unlabeled_indices = df[unlabeled_mask].index.tolist()

    print(f"总样本数: {total}")
    print(f"剩余待标注: {len(unlabeled_indices)}")
    print("-" * 40)
    print("操作说明:")
    print("  [1] : 标记为 球 (TP)")
    print("  [0] : 标记为 非球 (FP)")
    print("  [Space] : 跳过当前")
    print("  [B] : 返回上一张 (Back)")
    print("  [Q] : 保存并退出")
    print("-" * 40)

    # 设置起始位置（start_from 是在全量序列上的位置偏移）
    current_idx_ptr = max(0, min(start_from, len(full_indices) - 1)) if full_indices else 0
    if start_from > 0:
        print(f"游标已跳转到第 {current_idx_ptr + 1} 张（可用 p/n 前后切换，或 B 回到上一步）")
    
    history_stack = [] # 记录访问过的 index，用于回退

    while True:
        # 如果都标注完了则结束
        unlabeled_mask = df["label"].isna() | (df["label"].astype(str).str.strip() == "")
        remaining = unlabeled_mask.sum()
        if remaining == 0:
            print("所有图片已标注完成！")
            break

        # 从全量索引中取当前样本（支持查看之前/之后任意图片）
        if current_idx_ptr < 0:
            current_idx_ptr = 0
        if current_idx_ptr >= len(full_indices):
            current_idx_ptr = len(full_indices) - 1
        idx = full_indices[current_idx_ptr]
        row = df.loc[idx]
        
        # 构造图片路径（兼容多种运行目录）
        patch_rel_path = row["patch_path"]
        found_path = None

        # 候选路径（按优先级尝试）
        candidates = []
        # 1) 如果是绝对路径，直接尝试
        if osp.isabs(patch_rel_path):
            candidates.append(patch_rel_path)
        # 2) 直接按相对路径（以当前工作目录）
        candidates.append(osp.abspath(patch_rel_path))
        # 3) 以 manifest 所在目录拼接（常见情况）
        candidates.append(osp.join(manifest_dir, patch_rel_path))
        # 4) 规范化可能重复的路径（避免 manifest_dir/.../manifest_dir/... 的情况）
        candidates.append(osp.normpath(osp.join(manifest_dir, osp.basename(patch_rel_path))))

        for cand in candidates:
            if osp.isfile(cand):
                found_path = cand
                break

        if found_path is None:
            # 输出尝试过的路径，便于调试
            tried = ", ".join(candidates[:3])
            print(f"[警告] 图片不存在，跳过: {osp.join(manifest_dir, patch_rel_path)}")
            print(f"       已尝试: {tried}")
            current_idx_ptr += 1
            continue

        patch_full_path = found_path
            
        img = cv2.imread(patch_full_path)
        if img is None:
            print(f"[警告] 图片无法读取，跳过: {patch_full_path}")
            current_idx_ptr += 1
            continue

        # 放大图片以便查看
        img_display = cv2.resize(img, (patch_size_display, patch_size_display), interpolation=cv2.INTER_NEAREST)
        
        # 获取当前标签状态
        current_label = row.get("label", "")
        if pd.isna(current_label) or str(current_label).strip() == "":
            label_status = "None"
            label_color = (200, 200, 200)  # 灰色
        elif int(current_label) == 1:
            label_status = "Ball (1)"
            label_color = (0, 255, 0)  # 绿色
        else:
            label_status = "Background (0)"
            label_color = (0, 0, 255)  # 红色
        
        # 在图片上绘制信息（显示在全量序列中的位置，以及该样本是否已被标注）
        info_text = f"Pos: {current_idx_ptr+1}/{len(full_indices)}  (index: {idx})"
        cv2.putText(img_display, info_text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        label_text = f"Label: {label_status}"
        cv2.putText(img_display, label_text, (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2)
        cv2.putText(img_display, "1:Ball, 0:Bg, p:Prev, n:Next, B:Back, Q:Quit", (5, patch_size_display - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        cv2.imshow("Labeling Tool", img_display)
        
        key = cv2.waitKey(0)

        # 按键处理
        if key == ord('q') or key == 27: # q or ESC
            print("正在保存并退出...")
            break
            
        elif key == ord('1'):
            df.at[idx, "label"] = 1
            print(f"[{idx}] marked as BALL (1)")
            # 立即保存
            df.to_csv(manifest_path, index=False)
            history_stack.append(current_idx_ptr)
            current_idx_ptr += 1
            
        elif key == ord('0'):
            df.at[idx, "label"] = 0
            print(f"[{idx}] marked as BACKGROUND (0)")
            # 立即保存
            df.to_csv(manifest_path, index=False)
            history_stack.append(current_idx_ptr)
            current_idx_ptr += 1
            
        elif key == ord(' '): # Space to skip
            print(f"[{idx}] Skipped")
            history_stack.append(current_idx_ptr)
            current_idx_ptr += 1
        elif key == ord('n') or key == ord('N'):
            # 向后（下一个）查看
            history_stack.append(current_idx_ptr)
            current_idx_ptr += 1
            continue
        elif key == ord('p') or key == ord('P'):
            # 向前（上一个）查看
            history_stack.append(current_idx_ptr)
            current_idx_ptr = max(0, current_idx_ptr - 1)
            continue
        elif key == ord('b') or key == ord('B'): # Back
            if len(history_stack) > 0:
                prev_ptr = history_stack.pop()
                current_idx_ptr = prev_ptr
                print(f"返回上一张（位置 {current_idx_ptr+1}）")
            else:
                print("历史记录为空，无法回退。可使用 'p' 查看前一张。")
        else:
            print("无效按键，请按 1, 0, Q, Space 或 B")

        # 每 10 次操作自动保存一次，防止崩溃丢失
        if len(history_stack) % 10 == 0:
             df.to_csv(manifest_path, index=False)

    # 最终保存
    df.to_csv(manifest_path, index=False)
    print(f"保存成功: {manifest_path}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FP 过滤二分类：交互式标注工具")
    parser.add_argument("--manifest", required=True, help="由 extract_patches.py 生成的 manifest.csv 路径")
    parser.add_argument("--size", type=int, default=256, help="显示窗口中的图片大小（像素）")
    parser.add_argument("--start-from", type=int, default=0, help="从第几张图片开始标注（从0开始计数，默认0）")
    
    args = parser.parse_args()
    
    label_patches(args.manifest, args.size, args.start_from)