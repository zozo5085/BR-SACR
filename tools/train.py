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
from model.model import SACR
from utils.test_mIoU import mean_iou
from utils.preprocess import val_preprocess, preprocess, read_file_list, prepare_dataset_cls_tokens

ssl._create_default_https_context = ssl._create_unverified_context


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def custom_collate_fn(batch):
    imgs, labels, metas, filenames, pseudo_classes = zip(*batch)
    imgs = torch.stack(imgs)
    labels = torch.stack(labels)
    return imgs, labels, metas, filenames, pseudo_classes


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', default='config/voc_train_ori_cfg.yaml', type=str)
    parser.add_argument('--exp_name', type=str, default='sacr_voc_train')
    parser.add_argument('--workers', type=int, default=-1)
    parser.add_argument('--init_path', type=str, default='')
    return parser.parse_args()


class TrainDataset(Dataset):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        (
            _,
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
        return model

    if not os.path.isfile(init_path):
        raise FileNotFoundError(f"Cannot find init checkpoint: {init_path}")

    ckpt = torch.load(init_path, map_location='cpu', weights_only=False)
    state_dict = clean_state_dict(ckpt)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    bad_missing = [k for k in missing if not is_allowed_missing_key(k)]
    bad_unexpected = [k for k in unexpected if not k.startswith('clip.')]

    if len(bad_missing) > 0:
        raise RuntimeError(f"Bad missing keys: {bad_missing}")

    if len(bad_unexpected) > 0:
        raise RuntimeError(f"Bad unexpected keys: {bad_unexpected}")

    print(f"[INFO] Loaded init checkpoint: {init_path}")
    return model


def set_trainable_sacr(model, cfg):
    for _, param in model.named_parameters():
        param.requires_grad = False

    base_trainable_names = {
        "text_encoder.prompt_token",
        "pe_proj.weight",
        "pe_proj.bias",
        "decoder_conv2.weight",
        "decoder_conv2.bias",
        "decoder_norm2.weight",
        "decoder_norm2.bias",
    }

    train_context_refine = bool(getattr(cfg.MODEL, "TRAIN_CONTEXT_REFINE", False))
    train_edge_gate = bool(getattr(cfg.MODEL, "TRAIN_EDGE_GATE", False))

    for name, param in model.named_parameters():
        if name in base_trainable_names:
            param.requires_grad = True

        if train_context_refine and name.startswith("context_refine."):
            param.requires_grad = True

        if train_edge_gate and name.startswith("edge_gate."):
            param.requires_grad = True

    return model


def check_trainable_params(model, cfg):
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    total_trainable = sum(p.numel() for _, p in trainable)

    expected_trainable = {
        "text_encoder.prompt_token",
        "pe_proj.weight",
        "pe_proj.bias",
        "decoder_conv2.weight",
        "decoder_conv2.bias",
        "decoder_norm2.weight",
        "decoder_norm2.bias",
    }

    train_context_refine = bool(getattr(cfg.MODEL, "TRAIN_CONTEXT_REFINE", False))
    train_edge_gate = bool(getattr(cfg.MODEL, "TRAIN_EDGE_GATE", False))

    if train_context_refine:
        for n, _ in model.named_parameters():
            if n.startswith("context_refine."):
                expected_trainable.add(n)

    if train_edge_gate:
        for n, _ in model.named_parameters():
            if n.startswith("edge_gate."):
                expected_trainable.add(n)

    actual_trainable = {n for n, _ in trainable}

    if actual_trainable != expected_trainable:
        raise RuntimeError(
            f"Trainable params mismatch.\n"
            f"Expected: {sorted(expected_trainable)}\n"
            f"Actual: {sorted(actual_trainable)}"
        )

    print(f"[INFO] Trainable parameter groups: {len(trainable)}")
    print(f"[INFO] Trainable parameters: {total_trainable:,}")


def train_single_gpu():
    set_seed(42)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    args = get_parser()
    cfg = cfg_from_file(args.cfg_file)

    exp_save_dir = os.path.join(cfg.SAVE_DIR, args.exp_name)
    os.makedirs(exp_save_dir, exist_ok=True)
    os.makedirs("experiments", exist_ok=True)

    log_path = os.path.join('experiments', f'log_{args.exp_name}.txt')
    log = open(log_path, mode='a', encoding='utf-8')

    workers = get_effective_workers(cfg, args)

    (
        _,
        val_filenames,
        train_images,
        _,
        val_images,
        val_labels,
        _,
        pseudo_classes
    ) = read_file_list(cfg)

    if len(pseudo_classes) != len(train_images):
        raise RuntimeError(
            f"Pseudo-label length mismatch: {len(pseudo_classes)} vs {len(train_images)}"
        )

    cls_name_token, classes = prepare_dataset_cls_tokens(cfg)
    text_weight = torch.load(cfg.DATASET.TEXT_WEIGHT, map_location="cpu").to(device)

    print("[INFO]  training")
    print(f"[INFO] config: {args.cfg_file}")
    print(f"[INFO] experiment: {args.exp_name}")
    print(f"[INFO] save_dir: {exp_save_dir}")
    print(f"[INFO] dataset: {cfg.DATASET.NAME}")
    print(f"[INFO] train images: {len(train_images)}")
    print(f"[INFO] val images: {len(val_images)}")
    print(f"[INFO] classes: {len(classes)}")

    print("[INFO]  training", file=log)
    print(f"[INFO] config: {args.cfg_file}", file=log)
    print(f"[INFO] experiment: {args.exp_name}", file=log)
    print(f"[INFO] save_dir: {exp_save_dir}", file=log)
    print(f"[INFO] dataset: {cfg.DATASET.NAME}", file=log)
    print(f"[INFO] train images: {len(train_images)}", file=log)
    print(f"[INFO] val images: {len(val_images)}", file=log)
    print(f"[INFO] classes: {len(classes)}", file=log)
    log.flush()

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

    clip_model, _ = clip.load("ViT-B/16")
    clip_model = clip_model.to(device)

    model = SACR(
        cfg=cfg,
        clip_model=clip_model,
        rank=device,
        zeroshot_weights=text_weight
    ).to(device)

    model = load_init_checkpoint(model, args.init_path)
    model = model.to(device)

    model = set_trainable_sacr(model, cfg)
    check_trainable_params(model, cfg)

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
            model.train()

            running_loss = 0.0
            valid_train_steps = 0
            skipped_empty_gt = 0
            skipped_bad_loss = 0

            lr = adjust_learning_rate_poly(
                optimizer,
                epoch,
                max_epoch,
                cfg.TRAIN.LR,
                power=0.9
            )

            loop = tqdm(train_loader, desc=f"Epoch {epoch} Training")

            for _, (img, label, img_metas, filenames, pseudo_class) in enumerate(loop):
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
                    print(f"[WARN] Non-finite loss at epoch={epoch}. Skip batch.")
                    continue

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                valid_train_steps += 1

                avg_loss = running_loss / max(valid_train_steps, 1)

                loop.set_postfix(
                    lr=f"{lr:.6f}",
                    loss=f"{loss.item():.4f}",
                    avg=f"{avg_loss:.4f}"
                )

            epoch_avg_loss = running_loss / max(valid_train_steps, 1)

            model.eval()
            current_results_iou = []

            with torch.no_grad():
                for v_idx in tqdm(range(len(val_images)), desc="Validating"):
                    with open(val_images[v_idx], 'rb') as f:
                        value_buf = f.read()

                    img = val_preprocess(cfg, value_buf).unsqueeze(dim=0).to(device)

                    label = Image.open(val_labels[v_idx])
                    ori_shape = tuple((label.size[1], label.size[0]))
                    shape = img.shape[2:]

                    output = model(
                        img,
                        [],
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

                    save_path = os.path.join(exp_save_dir, val_filenames[v_idx] + '.pt')
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save(output.cpu(), save_path)
                    current_results_iou.append(save_path)

                iou = mean_iou(
                    current_results_iou,
                    val_labels[:len(current_results_iou)],
                    num_classes=c_num + 1,
                    ignore_index=255,
                    nan_to_num=0,
                    reduce_zero_label=cfg.DATASET.REDUCE_ZERO_LABEL
                )

                avg = iou['IoU'][:c_num].sum() / c_num
                is_best = avg > best_iou

                if is_best:
                    best_iou = avg
                    torch.save(
                        model.state_dict(),
                        os.path.join(exp_save_dir, 'best_weight.pth')
                    )

                checkpoint_state = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_iou': best_iou,
                }

                save_checkpoint(
                    checkpoint_state,
                    exp_save_dir,
                    filename='latest_checkpoint.pth'
                )

                if is_best:
                    save_checkpoint(
                        checkpoint_state,
                        exp_save_dir,
                        filename='model_best.pth'
                    )

            msg = (
                f"Epoch {epoch}: "
                f"lr={lr:.6f}, "
                f"loss={epoch_avg_loss:.6f}, "
                f"mIoU={avg:.4f}, "
                f"best={best_iou:.4f}, "
                f"valid_steps={valid_train_steps}, "
                f"skipped_empty={skipped_empty_gt}, "
                f"skipped_bad={skipped_bad_loss}"
            )

            print(msg)
            print(msg, file=log)
            log.flush()

            if epoch == stop_epoch:
                break

    finally:
        log.close()


if __name__ == '__main__':
    train_single_gpu()
