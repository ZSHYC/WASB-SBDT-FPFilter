"""
第二步：使用标注好的 manifest（含 label 列）训练二分类模型，用于筛除 FP。

使用示例（在 fp_filter 目录下执行）：cd fp_filter(别忘了！)
cd fp_filter
python train_fp_filter.py ^
  --manifest patch_outputs/patches_train1/manifest.csv ^
            patch_outputs/patches_train2/manifest.csv ^
            patch_outputs/patches_train3/manifest.csv ^
            patch_outputs/patches_train4/manifest.csv ^
            patch_outputs/patches_train5/manifest.csv ^
            patch_outputs/patches_train6_right/manifest.csv ^
            patch_outputs/patches_train7_right/manifest.csv ^
            patch_outputs/patches_train8_right/manifest.csv ^
            patch_outputs/patches_train9_right/manifest.csv ^
            patch_outputs/patches_train10_right/manifest.csv ^
  --out-dir patch_outputs/model_resnet ^
  --epochs 50
  
python train_fp_filter.py --manifest patch_outputs/patches_train1/manifest.csv patch_outputs/patches_train2/manifest.csv patch_outputs/patches_train3/manifest.csv patch_outputs/patches_train4/manifest.csv patch_outputs/patches_train5/manifest.csv patch_outputs/patches_train6_right/manifest.csv patch_outputs/patches_train7_right/manifest.csv patch_outputs/patches_train8_right/manifest.csv patch_outputs/patches_train9_right/manifest.csv patch_outputs/patches_train10_right/manifest.csv --out-dir patch_outputs/model_resnet --epochs 80
"""
import os
import sys
import os.path as osp
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 保证从项目根或 src 运行都能找到 fp_filter（在 import 前执行）
_src = osp.normpath(osp.join(osp.dirname(osp.abspath(__file__)), ".."))
if _src not in sys.path:
    sys.path.insert(0, _src)

from fp_filter.dataset import PatchDataset, get_default_transform
from fp_filter.model import build_model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return total_loss / max(len(loader), 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    # 统计混淆矩阵：TP, TN, FP, FN (假设 1=球, 0=非球)
    tp, tn, fp, fn = 0, 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        
        # 计算混淆矩阵
        for p, t in zip(pred, y):
            if t == 1 and p == 1:
                tp += 1
            elif t == 0 and p == 0:
                tn += 1
            elif t == 0 and p == 1:
                fp += 1
            elif t == 1 and p == 0:
                fn += 1
    
    acc = correct / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    
    return {
        'loss': total_loss / max(len(loader), 1),
        'acc': acc,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def main():
    parser = argparse.ArgumentParser(description="训练 FP 过滤二分类模型")
    parser.add_argument("--manifest", "-m", required=True, nargs='+', help="已标注 label 的 manifest.csv 路径（可传多个）")
    parser.add_argument("--out-dir", "-o", default="./outputs/fp_filter", help="保存 checkpoint 与日志的目录")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例，默认 0.2")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=64, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--patch-size", type=int, default=128, help="与第一步提取的 patch 尺寸一致，仅用于 transform")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform_train = get_default_transform(is_train=True, patch_size=args.patch_size)
    transform_val = get_default_transform(is_train=False, patch_size=args.patch_size)

    # 支持多个 manifest 文件夹
    manifest_paths = args.manifest if isinstance(args.manifest, list) else [args.manifest]
    print(f"加载 {len(manifest_paths)} 个训练数据集:")
    for mp in manifest_paths:
        print(f"  - {mp}")
    
    # 加载所有数据集并合并
    all_datasets = [PatchDataset(mp, transform=None, target_one_hot=False) for mp in manifest_paths]
    full_dataset = torch.utils.data.ConcatDataset(all_datasets)
    n = len(full_dataset)
    if n == 0:
        raise ValueError("manifest 中没有已标注的样本，请先在 manifest 的 label 列填 1(球) 或 0(非球)")

    n_val = max(1, int(n * args.val_ratio))
    n_train = n - n_val
    indices = np.random.permutation(n)
    train_idx, val_idx = indices[:n_train], indices[n_train:]
    # 为训练和验证分别创建带 transform 的合并数据集
    train_datasets = [PatchDataset(mp, transform=transform_train, target_one_hot=False) for mp in manifest_paths]
    val_datasets = [PatchDataset(mp, transform=transform_val, target_one_hot=False) for mp in manifest_paths]
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    val_dataset = torch.utils.data.ConcatDataset(val_datasets)
    train_sub = torch.utils.data.Subset(train_dataset, train_idx.tolist())
    val_sub = torch.utils.data.Subset(val_dataset, val_idx.tolist())

    train_loader = DataLoader(train_sub, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_sub, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ----------------------------------------------------
    # 新增：计算类别权重以解决样本不平衡问题 (High FP / Low TN)
    # ----------------------------------------------------
    # 从所有数据集中统计标签
    all_labels = []
    for ds in all_datasets:
        all_labels.extend(ds.df["label"].values.tolist())
    all_labels = np.array(all_labels)
    n_neg = (all_labels == 0).sum()
    n_pos = (all_labels == 1).sum()
    print(f"Dataset stats: Total={n}, Pos(1)={n_pos}, Neg(0)={n_neg}")
    if n_neg == 0 or n_pos == 0:
        print("警告: 某一类样本数为0，无法应用类别权重。")
        class_weights = None
    else:
        # 权重计算公式: Total / (NumClasses * Count)
        w0 = n / (2.0 * n_neg)
        w1 = n / (2.0 * n_pos)
        class_weights = torch.tensor([w0, w1], dtype=torch.float).to(device)
        print(f"Applying Class Weights: Neg(0)={w0:.4f}, Pos(1)={w1:.4f}")

    model = build_model().to(device)
    # 将计算出的权重传入 Loss
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Some PyTorch versions don't accept the `verbose` kwarg here; omit it for compatibility.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    best_f1 = 0.0
    history = {
        "train_loss": [], "train_acc": [], "train_precision": [], "train_recall": [], "train_f1": [],
        "val_loss": [], "val_acc": [], "val_precision": [], "val_recall": [], "val_f1": [],
        "val_tp": [], "val_tn": [], "val_fp": [], "val_fn": []
    }

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_metrics['f1'])

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_metrics['loss'])
        history["val_acc"].append(val_metrics['acc'])
        history["val_precision"].append(val_metrics['precision'])
        history["val_recall"].append(val_metrics['recall'])
        history["val_f1"].append(val_metrics['f1'])
        history["val_tp"].append(val_metrics['tp'])
        history["val_tn"].append(val_metrics['tn'])
        history["val_fp"].append(val_metrics['fp'])
        history["val_fn"].append(val_metrics['fn'])
        # 训练集指标暂时用占位符（避免增加计算开销，如需要可改为调用 evaluate）
        history["train_precision"].append(0.0)
        history["train_recall"].append(0.0)
        history["train_f1"].append(0.0)

        print(f"Epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f} "
              f"val_p={val_metrics['precision']:.4f} val_r={val_metrics['recall']:.4f} val_f1={val_metrics['f1']:.4f}")
        print(f"  TP={val_metrics['tp']} TN={val_metrics['tn']} FP={val_metrics['fp']} FN={val_metrics['fn']}")

        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            ckpt_path = osp.join(args.out_dir, "best.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_metrics['f1'],
                "val_acc": val_metrics['acc'],
                "val_metrics": val_metrics,
                "args": vars(args),
            }, ckpt_path)
            print(f"  保存最佳模型 (f1={best_f1:.4f}): {ckpt_path}")

        # 保存本轮模型
        epoch_ckpt_path = osp.join(args.out_dir, f"epoch_{epoch}.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_f1": val_metrics['f1'],
            "val_acc": val_metrics['acc'],
            "val_metrics": val_metrics,
            "args": vars(args),
        }, epoch_ckpt_path)

    torch.save({"epoch": args.epochs, "model_state_dict": model.state_dict(), "args": vars(args)},
               osp.join(args.out_dir, "last.pth"))
    with open(osp.join(args.out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"训练完成，最佳验证 F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
