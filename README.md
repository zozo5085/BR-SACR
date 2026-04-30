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


```text
├── datasets
│   ├── PASCAL VOC
│   │   ├── VOC2012
│   │   │   ├── JPEGImages
│   │   │   ├── SegmentationClass
│   │   │   ├── ImageSets
│   │   │   │   ├── Segmentation
│   │   ├── VOC2010
│   │   │   ├── JPEGImages
│   │   │   ├── SegmentationClassContext
│   │   │   ├── ImageSets
│   │   │   │   ├── SegmentationContext
│   │   │   │   │   ├── train.txt
│   │   │   │   │   ├── val.txt
│   │   │   ├── trainval_merged.json
│   ├── ADEChallengeData2016
│   │   ├── annotations
│   │   │   ├── training
│   │   │   ├── validation
│   │   ├── images
│   │   │   ├── training
│   │   │   ├── validation
│   ├── Cityscapes
│   │   ├── gtFine
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── leftImg8bit
│   │   │   ├── train
│   │   │   ├── val
│   ├── coco_stuff164k
│   │   ├── images
│   │   │   ├── train2017
│   │   │   ├── val2017
│   │   ├── annotations
│   │   │   ├── train2017
│   │   │   ├── val2017
```

You can also modify the dataset paths in the config files under config/.

For example, in config/voc_train_ori_cfg.yaml, set:

```bash
text/voc_ViT16_clip_text.pth
Available class sets include:
voc, context, ade, city, stuff
```
```YAML
DATASET:
  DATAROOT: 'your/path/to/VOC2012/'
  TEXT_WEIGHT: 'text/voc_ViT16_clip_text.pth'
```

# Training
Training on Pascal VOC:
```Bash
python tools/train.py --cfg config/voc_train_ori_cfg.yaml 
# Options for dataset: voc, context, ade, cityscapes, coco
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

