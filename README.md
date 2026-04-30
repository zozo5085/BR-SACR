# Robust CLIP-based Unsupervised Semantic Segmentation via Structure-Aware Bias Rectification

This repository provides the PyTorch implementation of our 2026 ICCE-TW paper:

**Robust CLIP-based Unsupervised Semantic Segmentation via Structure-Aware Bias Rectification**

We use **Python 3.9+** and **PyTorch 2.0+**.

# Installation

Clone this repository and install the required packages:

```bash
pip install -r requirements.txt
```

# Data Preparation

Please organize your datasets, such as PASCAL VOC, Cityscapes, ADE20K, or COCO-Stuff, under your local dataset directory.

You can also modify the dataset paths in the config files under config/.

For example, in config/voc_train_ori_cfg.yaml, set:

```bash
text/voc_ViT16_clip_text.pth
``
Available class sets include:
```bash
voc, context, ade, city, stuff
```
```YAML
DATASET:
  DATAROOT: 'your/path/to/VOC2012/'
  TEXT_WEIGHT: 'text/voc_ViT16_clip_text.pth'
```



# Evaluation
Evaluate a trained model on the validation set:
```bash
python tools/test.py --cfg config/voc_test_ori_cfg.yaml
```

if needed, set the checkpoint path in the corresponding config file:
```YAML
LOAD_PATH: 'path/to/bestweight.pth
```

#
