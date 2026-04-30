import sys
import os
import argparse
import torch
import clip
import torch.nn.functional as F
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.configs import cfg_from_file
from model.model import RECLIPPP
from utils.test_mIoU import mean_iou
from utils.preprocess import val_preprocess, read_file_list, prepare_dataset_cls_tokens


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
clip_model, _ = clip.load("ViT-B/16")
clip_model = clip_model.to(device)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', default='config/voc_test_ori_cfg.yaml', type=str)
    parser.add_argument('--model', dest='model_name', default='RECLIPPP', choices=['RECLIPPP'], type=str)
    parser.add_argument('--strict', action='store_true', help='use strict=True when loading checkpoint')
    return parser.parse_args()


def load_checkpoint(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def clean_state_dict(weight):
    if isinstance(weight, dict) and 'model_state_dict' in weight:
        weight = weight['model_state_dict']

    new_weight = {}
    for key, value in weight.items():
        if key.startswith('module.'):
            key = key[7:]
        new_weight[key] = value
    return new_weight


def check_critical_keys(model, missing, unexpected):
    critical = [
        "text_encoder.prompt_token",
        "pe_proj.weight",
        "pe_proj.bias",
        "decoder_conv2.weight",
        "decoder_conv2.bias",
        "decoder_norm2.weight",
        "decoder_norm2.bias",
        "context_refine.ca.conv1.weight",
        "context_refine.aspp.project.weight",
        "context_refine.fuse.weight",
        "edge_gate.alpha",
    ]

    miss_critical = [k for k in critical if k in missing]
    if len(miss_critical) > 0:
        raise RuntimeError(f"Critical learned keys missing: {miss_critical}")

    bad_unexpected = [k for k in unexpected if not k.startswith("clip.")]
    if len(bad_unexpected) > 0:
        print("[WARN] unexpected non-clip keys:", bad_unexpected[:30])

    print("[Load] missing keys:", len(missing))
    print("[Load] unexpected keys:", len(unexpected))
    if hasattr(model, "text_encoder"):
        print("prompt norm =", model.text_encoder.prompt_token.norm().item())
    if hasattr(model, "pe_proj"):
        print("pe_proj norm =", model.pe_proj.weight.norm().item())
    if hasattr(model, "decoder_conv2"):
        print("decoder norm =", model.decoder_conv2.weight.norm().item())


def test():
    args = get_parser()
    cfg = cfg_from_file(args.cfg_file)

    _, val_filenames, _, _, val_images, val_labels, results_iou, _ = read_file_list(cfg)
    cls_name_token, _ = prepare_dataset_cls_tokens(cfg)
    text_weight = torch.load(cfg.DATASET.TEXT_WEIGHT, map_location='cpu').to(device)

    model = RECLIPPP(cfg=cfg, clip_model=clip_model, rank=device, zeroshot_weights=text_weight)

    weight = load_checkpoint(cfg.LOAD_PATH)
    new_weight = clean_state_dict(weight)

    if args.strict:
        model.load_state_dict(new_weight, strict=True)
        missing, unexpected = [], []
    else:
        missing, unexpected = model.load_state_dict(new_weight, strict=False)

    print("CFG_FILE =", args.cfg_file)
    print("LOAD_PATH =", cfg.LOAD_PATH)
    print("TEST.PD =", cfg.TEST.PD if hasattr(cfg.TEST, "PD") else None)
    check_critical_keys(model, missing, unexpected)

    model = model.to(device)
    model.eval()

    c_num = cfg.DATASET.NUM_CLASSES

    with torch.no_grad():
        idx = 0
        for idx in range(len(val_images)):
            with open(val_images[idx], 'rb') as f:
                value_buf = f.read()

            img = val_preprocess(cfg, value_buf).unsqueeze(dim=0).to(device)

            label = Image.open(val_labels[idx])
            ori_shape = tuple((label.size[1], label.size[0]))
            shape = img.shape[2:]

            gt_cls = []
            output = model(img, gt_cls, text_weight, cls_name_token, training=False)

            # official-style PD
            N, C, H, W = output.shape
            if args.model_name == 'RECLIPPP':
                _output = F.softmax(output * 10, dim=1)
                max_cls_conf = _output.view(N, C, -1).max(dim=-1)[0]
                selected_cls = (max_cls_conf < cfg.TEST.PD)[:, :, None, None].expand(N, C, H, W)
                output[selected_cls] = -100
            else:
                _output = F.softmax(output * 100, dim=1)
                max_cls_conf = _output.view(N, C, -1).max(dim=-1)[0]
                selected_cls = (max_cls_conf < cfg.TEST.ReCLIP_PD)[:, :, None, None].expand(N, C, H, W)
                output[selected_cls] = -100

            output = F.interpolate(output, shape, None, 'bilinear', False).reshape(1, c_num, shape[0], shape[1])
            output = F.interpolate(output, ori_shape, None, 'bilinear', False).reshape(1, c_num, ori_shape[0], ori_shape[1])
            output = F.softmax(output, dim=1)
            output = torch.argmax(output, dim=1).squeeze(dim=0)

            save_path = os.path.join(cfg.SAVE_DIR, val_filenames[idx] + '.pt')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(output.cpu(), save_path)

            if idx % 100 == 0:
                print(f'filenames:{val_filenames[idx]}, img_idx:{idx}')

        iou = mean_iou(
            results_iou,
            val_labels,
            num_classes=c_num + 1,
            ignore_index=255,
            nan_to_num=0,
            reduce_zero_label=cfg.DATASET.REDUCE_ZERO_LABEL
        )
        print(iou['IoU'])
        avg = iou['IoU'].sum() / c_num
        print('avg:%.4f' % avg)
        print('\n\nfinish with %d/%d\nthe mIOU:%.4lf' % (idx, len(val_images), avg))


if __name__ == '__main__':
    test()
