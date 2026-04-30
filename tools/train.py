import sys
import os
import argparse
import ssl
import random
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import clip
from torch import optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.configs import cfg_from_file
from model.model import RECLIPPP
from utils.test_mIoU import mean_iou
from utils.preprocess import val_preprocess, preprocess, read_file_list, prepare_dataset_cls_tokens

ssl._create_default_https_context = ssl._create_unverified_context


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 訓練穩定優先
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def custom_collate_fn(batch):
    imgs, labels, metas, filenames, pseudo_classes = zip(*batch)
    imgs = torch.stack(imgs)
    labels = torch.stack(labels)
    return imgs, labels, metas, filenames, pseudo_classes


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        default='config/voc_train_ori_cfg.yaml',
        type=str
    )

    parser.add_argument(
        '--exp_name',
        type=str,
        default='reclippp_voc_train'
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=-1
    )

    # debug 用，不給就是正常完整訓練
    parser.add_argument(
        '--debug_train_batches',
        type=int,
        default=-1
    )

    parser.add_argument(
        '--debug_val_batches',
        type=int,
        default=-1
    )

    parser.add_argument(
        '--debug_epochs',
        type=int,
        default=-1
    )

    parser.add_argument(
        '--show_pseudo_samples',
        type=int,
        default=8
    )

    parser.add_argument(
        '--save_debug_masks',
        action='store_true'
    )

    # 可選：如果你想從 baseline checkpoint 接著訓練，就填路徑
    # 不填就是從目前 model 初始化開始
    parser.add_argument(
        '--init_path',
        type=str,
        default='',
        help='optional checkpoint path for initialization'
    )

    return parser.parse_args()


class TrainDataset(Dataset):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        (
            self.train_filenames,
            _,
            self.train_images,
            self.train_labels,
            _,
            _,
            _,
            self.pseudo_classes
        ) = read_file_list(cfg)

    def __getitem__(self, idx):
        with open(self.train_images[idx], 'rb') as f:
            value_buf = f.read()

        with open(self.train_labels[idx], 'rb') as f:
            label_buf = f.read()

        img, label, img_metas = preprocess(
            self.cfg,
            value_buf,
            label_buf,
            return_meta=True,
            unlabeled=False
        )

        return img, label, img_metas, self.train_images[idx], self.pseudo_classes[idx]

    def __len__(self):
        return len(self.train_images)


def adjust_learning_rate_poly(optimizer, epoch, num_epochs, base_lr, power):
    lr = base_lr * (1 - epoch / num_epochs) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(state, save_dir, filename):
    filepath = os.path.join(save_dir, filename)
    torch.save(state, filepath)


def get_effective_workers(cfg, args):
    if args.workers >= 0:
        return args.workers
    return int(getattr(cfg, 'NUM_WORKERS', 0))


def summarize_pseudo_classes(pseudo_classes, num_classes):
    hist = torch.zeros(num_classes, dtype=torch.long)
    empty_cnt = 0
    max_len = 0

    for row in pseudo_classes:
        if row is None or len(row) == 0:
            empty_cnt += 1
            continue

        max_len = max(max_len, len(row))

        for c in row:
            c = int(c)
            if 0 <= c < num_classes:
                hist[c] += 1

    return hist, empty_cnt, max_len


def clean_state_dict(weight):
    if isinstance(weight, dict) and 'model_state_dict' in weight:
        weight = weight['model_state_dict']

    new_weight = {}
    for key, value in weight.items():
        if key.startswith('module.'):
            key = key[7:]
        new_weight[key] = value

    return new_weight


def is_allowed_missing_key(key):
    """
    用舊 checkpoint 載入目前 model 時，可能會少掉 optional BR-SACR module。
    這些 module 預設 disabled/frozen，所以允許 missing。
    """
    allowed_prefixes = [
        'context_refine.',
        'edge_gate.',
    ]

    for prefix in allowed_prefixes:
        if key.startswith(prefix):
            return True

    return False


def load_init_checkpoint(model, init_path):
    if init_path is None or init_path == '':
        print("[Init] No init checkpoint is used.")
        return model

    if not os.path.isfile(init_path):
        raise FileNotFoundError(f"[Init] Cannot find init checkpoint: {init_path}")

    print(f"[Init] Loading checkpoint: {init_path}")

    ckpt = torch.load(init_path, map_location='cpu', weights_only=False)
    state_dict = clean_state_dict(ckpt)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    bad_missing = [k for k in missing if not is_allowed_missing_key(k)]
    bad_unexpected = [k for k in unexpected if not k.startswith('clip.')]

    print("[Init] missing keys:", missing)
    print("[Init] unexpected keys:", unexpected)

    if len(bad_missing) > 0:
        raise RuntimeError(f"[Init] Bad missing keys: {bad_missing}")

    if len(bad_unexpected) > 0:
        raise RuntimeError(f"[Init] Bad unexpected keys: {bad_unexpected}")

    return model


def set_trainable_author_aligned(model):
    """
    對齊你原本正常版：
    只訓練 7 組參數。
    """
    for name, param in model.named_parameters():
        param.requires_grad = False

    trainable_names = {
        "text_encoder.prompt_token",
        "pe_proj.weight",
        "pe_proj.bias",
        "decoder_conv2.weight",
        "decoder_conv2.bias",
        "decoder_norm2.weight",
        "decoder_norm2.bias",
    }

    for name, param in model.named_parameters():
        if name in trainable_names:
            param.requires_grad = True

    return model


def check_trainable_params(model):
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    print(f"[Debug] Trainable params: {len(trainable)}")
    for n, p in trainable:
        print(f"  {n}: {tuple(p.shape)}")

    expected_trainable = {
        "text_encoder.prompt_token",
        "pe_proj.weight",
        "pe_proj.bias",
        "decoder_conv2.weight",
        "decoder_conv2.bias",
        "decoder_norm2.weight",
        "decoder_norm2.bias",
    }

    actual_trainable = {n for n, p in trainable}

    if actual_trainable != expected_trainable:
        raise RuntimeError(
            f"Trainable params mismatch.\n"
            f"Expected: {sorted(expected_trainable)}\n"
            f"Actual: {sorted(actual_trainable)}"
        )


def is_debug_mode(args):
    return (
        args.debug_train_batches > 0
        or args.debug_val_batches > 0
        or args.debug_epochs > 0
    )


def train_single_gpu():
    set_seed(42)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    clip_model, _ = clip.load("ViT-B/16")
    clip_model = clip_model.to(device)

    args = get_parser()
    cfg = cfg_from_file(args.cfg_file)

    debug_mode = is_debug_mode(args)

    exp_save_dir = os.path.join(cfg.SAVE_DIR, args.exp_name)
    os.makedirs(exp_save_dir, exist_ok=True)
    os.makedirs("experiments", exist_ok=True)

    log_path = os.path.join('experiments', f'log_{args.exp_name}.txt')
    log = open(log_path, mode='a', encoding='utf-8')

    workers = get_effective_workers(cfg, args)

    (
        train_filenames,
        val_filenames,
        train_images,
        train_labels,
        val_images,
        val_labels,
        _,
        pseudo_classes
    ) = read_file_list(cfg)

    cls_name_token, classes = prepare_dataset_cls_tokens(cfg)
    text_weight = torch.load(cfg.DATASET.TEXT_WEIGHT, map_location="cpu").to(device)

    print("[INFO] Author-aligned RECLIPPP training")
    print(f"[INFO] exp_name={args.exp_name}")
    print(f"[INFO] save_dir={exp_save_dir}")
    print(f"[INFO] workers={workers}")
    print(f"[INFO] debug_mode={debug_mode}")
    print(f"[INFO] init_path={args.init_path}")

    print("[INFO] Author-aligned RECLIPPP training", file=log)
    print(f"[INFO] exp_name={args.exp_name}", file=log)
    print(f"[INFO] save_dir={exp_save_dir}", file=log)
    print(f"[INFO] workers={workers}", file=log)
    print(f"[INFO] debug_mode={debug_mode}", file=log)
    print(f"[INFO] init_path={args.init_path}", file=log)
    log.flush()

    print("classes =", classes)
    print("num classes =", len(classes))
    print("cls_name_token shape =", cls_name_token.shape)
    print("text_weight shape =", text_weight.shape)
    print("=" * 100)
    print("[Sanity Check]")
    print("len(train_images) =", len(train_images))
    print("len(train_labels) =", len(train_labels))
    print("len(pseudo_classes) =", len(pseudo_classes))
    print("len(val_images) =", len(val_images))
    print("len(val_labels) =", len(val_labels))

    pseudo_hist, empty_pseudo_cnt, max_pseudo_len = summarize_pseudo_classes(
        pseudo_classes,
        cfg.DATASET.NUM_CLASSES
    )

    print("empty pseudo count =", empty_pseudo_cnt)
    print("max pseudo length =", max_pseudo_len)
    print("pseudo class histogram =", pseudo_hist.tolist())

    n_show = min(args.show_pseudo_samples, len(train_images))
    for i in range(n_show):
        print(f"[pseudo sample {i}] {os.path.basename(train_images[i])} -> {pseudo_classes[i]}")

    print("=" * 100)

    if debug_mode:
        print("[WARN] Debug mode is enabled. mIoU is only for sanity check, not final performance.")

    train_data = TrainDataset(cfg)
    train_loader = DataLoader(
        dataset=train_data,
        shuffle=True,
        num_workers=workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(workers > 0),
        batch_size=cfg.TRAIN.BATCH_SIZE,
        collate_fn=custom_collate_fn
    )

    model = RECLIPPP(
        cfg=cfg,
        clip_model=clip_model,
        rank=device,
        zeroshot_weights=text_weight
    ).to(device)

    model = load_init_checkpoint(model, args.init_path)
    model = model.to(device)

    model = set_trainable_author_aligned(model)
    check_trainable_params(model)

    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.TRAIN.LR,
        momentum=0.9,
        weight_decay=0.0005
    )

    max_epoch = cfg.TRAIN.MAX_EPOCH
    stop_epoch = cfg.TRAIN.EPOCH if cfg.TRAIN.EPOCH >= 0 else max_epoch

    c_num = cfg.DATASET.NUM_CLASSES
    best_iou = 0.0

    try:
        for epoch in range(max_epoch):
            pt = model.text_encoder.prompt_token
            print(
                f"[Debug] epoch={epoch} "
                f"prompt_token norm={pt.norm().item():.6f} "
                f"grad={pt.grad is not None}"
            )

            model.train()

            running_loss = 0.0
            valid_train_steps = 0
            skipped_empty_gt = 0
            skipped_bad_loss = 0
            train_pred_hist = torch.zeros(c_num, dtype=torch.long)

            lr = adjust_learning_rate_poly(
                optimizer,
                epoch,
                max_epoch,
                cfg.TRAIN.LR,
                power=0.9
            )

            loop = tqdm(train_loader, desc=f"Epoch {epoch} Training")

            for idx, (img, label, img_metas, filenames, pseudo_class) in enumerate(loop):
                if args.debug_train_batches > 0 and idx >= args.debug_train_batches:
                    break

                gt_cls = [[int(t) for t in p_cls] for p_cls in pseudo_class]

                if sum(len(t) for t in gt_cls) == 0:
                    skipped_empty_gt += 1
                    continue

                img = img.to(device, non_blocking=True)

                output, loss = model(
                    img,
                    gt_cls,
                    text_weight,
                    cls_name_token,
                    training=True,
                    img_metas=img_metas
                )

                if not torch.isfinite(loss):
                    skipped_bad_loss += 1
                    print(f"[WARN] Non-finite loss at epoch={epoch}, iter={idx}. Skip batch.")
                    continue

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                valid_train_steps += 1

                with torch.no_grad():
                    batch_pred = torch.argmax(output.detach(), dim=1)
                    train_pred_hist += torch.bincount(
                        batch_pred.view(-1).cpu(),
                        minlength=c_num
                    )

                avg_loss = running_loss / max(valid_train_steps, 1)

                loop.set_postfix(
                    lr=f"{lr:.6f}",
                    loss=f"{loss.item():.4f}",
                    avg=f"{avg_loss:.4f}"
                )

            epoch_avg_loss = running_loss / max(valid_train_steps, 1)

            print(
                f'epoch {epoch} finish, lr:{lr:.6f}, avg_loss:{epoch_avg_loss:.6f}',
                file=log
            )
            print(
                f'[Train Hist] epoch={epoch} pred_hist={train_pred_hist.tolist()}',
                file=log
            )
            print(
                f'[Train Stat] epoch={epoch} valid_steps={valid_train_steps}, '
                f'skipped_empty_gt={skipped_empty_gt}, skipped_bad_loss={skipped_bad_loss}',
                file=log
            )
            log.flush()

            # =========================
            # Validation
            # No PD filter here.
            # softmax -> argmax directly.
            # =========================
            torch.backends.cudnn.benchmark = True
            model.eval()

            current_results_iou = []
            val_hist = torch.zeros(c_num, dtype=torch.long)

            with torch.no_grad():
                val_total = len(val_images)
                if args.debug_val_batches > 0:
                    val_total = min(len(val_images), args.debug_val_batches)

                for v_idx in tqdm(range(val_total), desc="Validating"):
                    with open(val_images[v_idx], 'rb') as f:
                        value_buf = f.read()

                    img = val_preprocess(cfg, value_buf).unsqueeze(dim=0).to(device)

                    label = Image.open(val_labels[v_idx])
                    ori_shape = tuple((label.size[1], label.size[0]))
                    shape = img.shape[2:]

                    gt_cls = []

                    output = model(
                        img,
                        gt_cls,
                        text_weight,
                        cls_name_token,
                        training=False
                    )

                    output = F.interpolate(
                        output,
                        shape,
                        None,
                        'bilinear',
                        False
                    ).reshape(1, c_num, shape[0], shape[1])

                    output = F.interpolate(
                        output,
                        ori_shape,
                        None,
                        'bilinear',
                        False
                    ).reshape(1, c_num, ori_shape[0], ori_shape[1])

                    output = F.softmax(output, dim=1)
                    output = torch.argmax(output, dim=1).squeeze(dim=0)

                    val_hist += torch.bincount(
                        output.view(-1).cpu(),
                        minlength=c_num
                    )

                    save_path = os.path.join(exp_save_dir, val_filenames[v_idx] + '.pt')
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)

                    torch.save(output.cpu(), save_path)
                    current_results_iou.append(save_path)

                    if args.save_debug_masks and v_idx < 3:
                        debug_mask_path = os.path.join(
                            exp_save_dir,
                            f"debug_epoch{epoch}_val{v_idx}.pt"
                        )
                        torch.save(output.cpu(), debug_mask_path)

                iou = mean_iou(
                    current_results_iou,
                    val_labels[:len(current_results_iou)],
                    num_classes=c_num + 1,
                    ignore_index=255,
                    nan_to_num=0,
                    reduce_zero_label=cfg.DATASET.REDUCE_ZERO_LABEL
                )

                # 只取前 c_num 類，避免 c_num+1 多出來的空類別影響
                avg = iou['IoU'][:c_num].sum() / c_num

                print(f"Epoch {epoch} strict mIoU: {avg:.4f}")
                print(f"[Val Hist] epoch={epoch} pred_hist={val_hist.tolist()}")

                print(f'epoch={epoch} strict_miou={avg:.4f}', file=log)
                print(f'[Val Hist] epoch={epoch} pred_hist={val_hist.tolist()}', file=log)
                log.flush()

                is_best = avg > best_iou

                if is_best:
                    best_iou = avg
                    print(f"*** New Best Strict Saved: {best_iou:.4f} ***")

                    if not debug_mode:
                        torch.save(
                            model.state_dict(),
                            os.path.join(exp_save_dir, 'best_weight.pth')
                        )

                if not debug_mode:
                    checkpoint_state = {
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_iou': best_iou,
                    }

                    save_checkpoint(
                        checkpoint_state,
                        exp_save_dir,
                        filename=f'checkpoint_epoch_{epoch}.pth'
                    )

                    if is_best:
                        save_checkpoint(
                            checkpoint_state,
                            exp_save_dir,
                            filename='model_best.pth'
                        )

            torch.backends.cudnn.benchmark = False

            if args.debug_epochs > 0 and (epoch + 1) >= args.debug_epochs:
                print(f"[Debug] stop after {args.debug_epochs} epoch(s).")
                break

            if epoch == stop_epoch:
                break

    finally:
        log.close()


if __name__ == '__main__':
    train_single_gpu()