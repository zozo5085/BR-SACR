import yaml
from easydict import EasyDict as edict

cfg = edict()

cfg.DATASET = edict()
cfg.DATASET.NAME = ''
cfg.DATASET.NUM_CLASSES = 0
cfg.DATASET.REDUCE_ZERO_LABEL = True
cfg.DATASET.DATAROOT = ''
cfg.DATASET.SCALE = []
cfg.DATASET.RATIO_RANGE = []
cfg.DATASET.CROP_SIZE = []
cfg.DATASET.CAT_MAX_RATIO = 0
cfg.DATASET.TEXT_WEIGHT = ''
cfg.DATASET.IMG_NORM_CFG = edict()
cfg.DATASET.IMG_NORM_CFG.MEAN = []
cfg.DATASET.IMG_NORM_CFG.STD = []
cfg.DATASET.IMG_NORM_CFG.RGB = True
cfg.DATASET.K = 0
cfg.DATASET.DISTILL_K = 0
cfg.DATASET.THRESHOLD = 0
cfg.DATASET.IGNORE_INDEX = 255
cfg.DATASET.PALETTE = []

cfg.MODEL = edict()
cfg.MODEL.FEATURE_EXTRACTOR = ''
cfg.MODEL.TEXT_CHANNEL = 0
cfg.MODEL.VISUAL_CHANNEL = 0
cfg.MODEL.TRAINING = False

# =========================================================
# BR-SACR switches
# 這些一定要先在 configs.py 註冊，
# 不然 yaml 裡面新增會被 merge_a_to_b 擋掉。
# =========================================================

# 是否在 forward 啟用 Context Refine
# Context Refine = Coordinate Attention + Zero-Init Mini-ASPP
cfg.MODEL.USE_CONTEXT_REFINE = False

# 是否訓練 Context Refine 的參數
cfg.MODEL.TRAIN_CONTEXT_REFINE = False

# 是否在 forward 啟用 Edge-Guided Residual Gating
cfg.MODEL.USE_EDGE_GATE = False

# 是否訓練 Edge Gate 的 alpha
cfg.MODEL.TRAIN_EDGE_GATE = False

# Edge Gate 的 alpha 初始值
# 0.0 代表一開始是 identity，不會破壞 baseline：
# x * (1 + 0 * edge) = x
cfg.MODEL.EDGE_ALPHA = 0.0

# 是否在 Bias Rectification 後套 ReLU
# 建議 False，對齊原本穩定版：
# S_rect = S_raw - lambda * B
cfg.MODEL.USE_RELU_RECT = False

# Bias Rectification 強度
cfg.MODEL.BIAS_LAMBDA = 1.0

# Coordinate Attention reduction ratio
cfg.MODEL.CA_REDUCTION = 32

# Mini-ASPP dilation rates
cfg.MODEL.ASPP_DILATIONS = [1, 2, 3]

cfg.TRAIN = edict()
cfg.TRAIN.BATCH_SIZE = 1
cfg.TRAIN.MAX_EPOCH = 50
cfg.TRAIN.EPOCH = 0
cfg.TRAIN.MAX_ITER = 0
cfg.TRAIN.LR = 0
cfg.TRAIN.LOG = ''

cfg.TEST = edict()
cfg.TEST.BATCH_SIZE = 0
cfg.TEST.PD = 0
cfg.TEST.ReCLIP_PD = 0.5

cfg.EVAL_METRIC = ''
cfg.SAVE_DIR = ''
cfg.NUM_WORKERS = 0
cfg.LOAD_PATH = ''
cfg.LOAD_DISTILL_PATH = ''


def merge_a_to_b(a, b):
    if type(a) is not edict:
        return
    for k in a:
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))
        if type(a[k]) is edict:
            merge_a_to_b(a[k], b[k])
        else:
            b[k] = a[k]
    return cfg


def cfg_from_file(filename):

    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f, Loader=yaml.FullLoader))
    merge_a_to_b(yaml_cfg, cfg)
    return cfg