import sys
import os
import argparse
import csv
from collections import defaultdict

import torch
import clip

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.configs import cfg_from_file
from model.model import RECLIPPP


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='config/voc_train_ori_cfg.yaml')
    parser.add_argument('--out', type=str, default='experiments/param_count.csv')
    parser.add_argument('--device', type=str, default='cuda:0')
    return parser.parse_args()


def count_params(params):
    return sum(p.numel() for p in params)


def fmt_num(x):
    return f"{x:,}"


def fmt_million(x):
    return f"{x / 1e6:.4f}M"


def get_top_module_name(name):
    return name.split('.')[0]


def main():
    args = get_parser()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    cfg = cfg_from_file(args.cfg)

    print("=" * 100)
    print("[Param Count]")
    print("cfg =", args.cfg)
    print("device =", device)
    print("=" * 100)

    clip_model, _ = clip.load("ViT-B/16")
    clip_model = clip_model.to(device)

    model = RECLIPPP(
        cfg=cfg,
        clip_model=clip_model,
        rank=device,
        zeroshot_weights=None
    ).to(device)

    total_params = count_params(model.parameters())
    trainable_params = count_params(p for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print("[Overall]")
    print("Total Params     :", fmt_num(total_params), f"({fmt_million(total_params)})")
    print("Trainable Params :", fmt_num(trainable_params), f"({fmt_million(trainable_params)})")
    print("Frozen Params    :", fmt_num(frozen_params), f"({fmt_million(frozen_params)})")
    print("Trainable Ratio  :", f"{100.0 * trainable_params / max(total_params, 1):.6f}%")
    print("=" * 100)

    print("[Trainable Parameter List]")
    trainable_list = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            n = p.numel()
            trainable_list.append((name, tuple(p.shape), n))
            print(f"{name:50s} {str(tuple(p.shape)):25s} {fmt_num(n)}")

    print("-" * 100)
    print("Trainable groups :", len(trainable_list))
    print("Trainable Params :", fmt_num(sum(x[2] for x in trainable_list)))
    print("=" * 100)

    module_stats = defaultdict(lambda: {"total": 0, "trainable": 0})

    for name, p in model.named_parameters():
        top = get_top_module_name(name)
        n = p.numel()
        module_stats[top]["total"] += n
        if p.requires_grad:
            module_stats[top]["trainable"] += n

    print("[By Top-Level Module]")
    rows = []
    for module_name, stat in sorted(module_stats.items()):
        total = stat["total"]
        trainable = stat["trainable"]
        rows.append({
            "module": module_name,
            "total_params": total,
            "trainable_params": trainable,
            "total_M": total / 1e6,
            "trainable_M": trainable / 1e6,
        })
        print(
            f"{module_name:25s} "
            f"total={fmt_num(total):>15s} ({fmt_million(total):>10s}) | "
            f"trainable={fmt_num(trainable):>15s} ({fmt_million(trainable):>10s})"
        )

    print("=" * 100)

    # Specifically count optional BR-SACR modules if they exist.
    extra_prefixes = [
        "context_refine",
        "edge_gate",
    ]

    extra_total = 0
    extra_trainable = 0

    for name, p in model.named_parameters():
        if any(name.startswith(prefix + ".") for prefix in extra_prefixes):
            extra_total += p.numel()
            if p.requires_grad:
                extra_trainable += p.numel()

    print("[Optional BR-SACR Added Modules]")
    print("Added Total Params     :", fmt_num(extra_total), f"({fmt_million(extra_total)})")
    print("Added Trainable Params :", fmt_num(extra_trainable), f"({fmt_million(extra_trainable)})")
    print("=" * 100)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "module",
                "total_params",
                "trainable_params",
                "total_M",
                "trainable_M",
            ]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[Saved] {args.out}")


if __name__ == "__main__":
    main()