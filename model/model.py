from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
import clip
import math
import cv2
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", nn.GELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.attn_mask = attn_mask

    def forward(self, x: torch.Tensor):
        x = x + self.attn(
            self.ln_1(x),
            self.ln_1(x),
            self.ln_1(x),
            need_weights=True,
            attn_mask=None
        )[0]
        x = x + self.mlp(self.ln_2(x))
        return x

    def _initialize_weights(self, clip_model, i):
        self.ln_1 = clip_model.visual.transformer.resblocks[i].ln_1
        self.ln_1.eps = 1e-06
        self.attn = clip_model.visual.transformer.resblocks[i].attn.to(torch.float32)
        self.attn.batch_first = True
        self.mlp = clip_model.visual.transformer.resblocks[i].mlp.to(torch.float32)
        self.ln_2 = clip_model.visual.transformer.resblocks[i].ln_2
        self.ln_2.eps = 1e-06

        for p in self.parameters():
            p.requires_grad = False


class LastResidualAttentionBlock(nn.Module):
    def __init__(self, clip_model: clip, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", nn.GELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask
        self._initialize_weights(clip_model)

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        y = self.ln_1(x)
        y = F.linear(y, self.attn.in_proj_weight, self.attn.in_proj_bias)

        N, L, C = y.shape
        y = y.view(N, L, 3, C // 3).permute(2, 0, 1, 3).reshape(3 * N, L, C // 3)
        y = F.linear(y, self.attn.out_proj.weight, self.attn.out_proj.bias)

        q, k, v = y.tensor_split(3, dim=0)

        v += x
        v = v + self.mlp(self.ln_2(v))

        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        return [x, q, k, v]

    def _initialize_weights(self, clip_model):
        self.ln_1 = clip_model.visual.transformer.resblocks[11].ln_1
        self.ln_1.eps = 1e-06
        self.attn = clip_model.visual.transformer.resblocks[11].attn.to(torch.float32)
        self.attn.batch_first = True
        self.mlp = clip_model.visual.transformer.resblocks[11].mlp.to(torch.float32)
        self.ln_2 = clip_model.visual.transformer.resblocks[11].ln_2
        self.ln_2.eps = 1e-06

        for p in self.parameters():
            p.requires_grad = False


class Transformer(nn.Module):
    def __init__(self, clip_model: clip, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblock = []

        for i in range(self.layers - 1):
            self.resblock.append(ResidualAttentionBlock(width, heads, attn_mask))

        self.resblock.append(LastResidualAttentionBlock(clip_model, width, heads, attn_mask))
        self._initialize_weights(clip_model)
        self.resblocks = nn.Sequential(*self.resblock)

    def forward(self, x: torch.Tensor):
        z, q, k, v = self.resblocks(x)
        return z, q, k, v

    def _initialize_weights(self, clip_model):
        for i in range(self.layers - 1):
            self.resblock[i]._initialize_weights(clip_model, i)


class VisionTransformer(nn.Module):
    def __init__(self, clip_model: clip, input_resolution: int, patch_size: int,
                 width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.patch_size = patch_size
        self.dilation = [1, 1]

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False
        )

        self.cls_token = torch.load('utils/cls_token.pt', map_location='cpu').to(torch.float32)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width)
        )
        self.ln_pre = nn.LayerNorm(width)

        self.transformer = Transformer(clip_model, width, layers, heads)
        self.ln_post = nn.LayerNorm(width)
        self.proj = clip_model.visual.proj.to(torch.float32)

        self._initialize_weights(clip_model)

    def forward(self, x, train=False, img_metas=None):
        B = x.shape[0]

        input_h, input_w = x.size()[-2:]
        kernel_h, kernel_w = (self.patch_size, self.patch_size)
        stride_h, stride_w = (self.patch_size, self.patch_size)
        output_h = math.ceil(input_h / stride_h)
        output_w = math.ceil(input_w / stride_w)

        pad_h = max((output_h - 1) * stride_h + (kernel_h - 1) * self.dilation[0] + 1 - input_h, 0)
        pad_w = max((output_w - 1) * stride_w + (kernel_w - 1) * self.dilation[1] + 1 - input_w, 0)

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [0, pad_w, 0, pad_h])

        x = x.to(device)
        x = self.conv1(x)
        x = x.flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(B, -1, -1).to(x.device)
        x = torch.cat((cls_tokens, x), dim=1)

        positional_embedding = self.positional_embedding.unsqueeze(dim=0)
        pos_h = self.input_resolution // self.patch_size
        pos_w = self.input_resolution // self.patch_size

        cls_token_weight = positional_embedding[:, 0]
        pos_embed_weight = positional_embedding[:, (-1 * pos_h * pos_w):]
        pos_embed_weight = pos_embed_weight.reshape(
            1, pos_h, pos_w, positional_embedding.shape[2]
        ).permute(0, 3, 1, 2)

        pos_embed_weight = F.interpolate(
            pos_embed_weight,
            size=(output_h, output_w),
            mode='bicubic',
            align_corners=False
        )

        cls_token_weight = cls_token_weight.unsqueeze(1)
        pos_embed_weight = torch.flatten(pos_embed_weight, 2).transpose(1, 2)
        positional_embedding = torch.cat((cls_token_weight, pos_embed_weight), dim=1)

        x = x + positional_embedding
        x = self.ln_pre(x)

        x, q, k, v = self.transformer(x)

        x = self.ln_post(x)
        v = self.ln_post(v)

        q = q[:, 1:]
        k = k[:, 1:]
        v = v[:, 1:]

        v = v.reshape(B, output_h, output_w, -1).permute(0, 3, 1, 2).contiguous()
        cls_token = x[:, 0]

        z_global = cls_token @ self.proj if self.proj is not None else None

        return [v, (output_h, output_w), z_global, k, positional_embedding[:, 1:, :]]

    def _initialize_weights(self, clip_model):
        self.conv1 = clip_model.visual.conv1.to(torch.float32)
        self.class_embedding = clip_model.visual.class_embedding
        self.positional_embedding = clip_model.visual.positional_embedding
        self.ln_pre = clip_model.visual.ln_pre
        self.ln_post = clip_model.visual.ln_post

        for p in self.parameters():
            p.requires_grad = False


class TextEncoder(nn.Module):
    def __init__(self, clip_model, training=False, cfg=None, device=None):
        super().__init__()
        self.transformer = clip_model.transformer.to(torch.float32)
        self.token_embedding = clip_model.token_embedding.to(torch.float32)
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection.to(torch.float32)
        self.dtype = torch.float32
        self.device = device

        token = torch.zeros((1, 73), dtype=torch.int).to(self.device)
        prompt_token = self.token_embedding(token)

        for p in self.parameters():
            p.requires_grad = False

        self.prompt_token = nn.Parameter(prompt_token)

        if (not training) and cfg is not None and hasattr(cfg, "LOAD_PATH") and cfg.LOAD_PATH:
            try:
                ckpt = torch.load(cfg.LOAD_PATH, map_location='cpu', weights_only=False)
                state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt

                prompt_token = None
                for key in [
                    'module.text_encoder.prompt_token',
                    'text_encoder.prompt_token'
                ]:
                    if isinstance(state_dict, dict) and key in state_dict:
                        prompt_token = state_dict[key]
                        break

                if prompt_token is not None:
                    self.prompt_token = nn.Parameter(prompt_token.to(torch.float32), requires_grad=False)
            except Exception as e:
                print(f"[WARN] TextEncoder prompt preload skipped: {e}")

    def forward(self, cls_name_token):
        device = self.device

        prompt_token = self.prompt_token.repeat(cls_name_token.shape[0], 1, 1).to(device)
        cls_name_token = cls_name_token.to(device)

        start_token = self.token_embedding(
            torch.tensor(49406, dtype=torch.int, device=device)
        ).repeat(cls_name_token.shape[0], 1, 1).to(device)

        cls_token = self.token_embedding(cls_name_token).to(device)

        x = torch.cat([start_token, prompt_token, cls_token], dim=1)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), 74 + cls_name_token.argmax(dim=-1)] @ self.text_projection
        return x

class CoordinateAttention(nn.Module):
    def __init__(self, channels, reduction=32):
        super().__init__()
        mip = max(8, channels // reduction)

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.conv1 = nn.Conv2d(channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)

        self.conv_h = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))

        return identity * a_h * a_w


class ZeroInitMiniASPP(nn.Module):
    def __init__(self, channels, hidden=None, dilations=(1, 2, 3)):
        super().__init__()
        hidden = hidden or max(16, channels)

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, hidden, kernel_size=3, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True)
            )
            for d in dilations
        ])

        self.project = nn.Conv2d(hidden * len(dilations), channels, kernel_size=1, bias=True)

        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, x):
        feats = [branch(x) for branch in self.branches]
        return self.project(torch.cat(feats, dim=1))


class StructureAwareContextRefinement(nn.Module):
    def __init__(self, channels, reduction=32, dilations=(1, 2, 3)):
        super().__init__()

        self.ca = CoordinateAttention(channels, reduction=reduction)
        self.aspp = ZeroInitMiniASPP(channels, channels, dilations=dilations)

        self.fuse = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=True)

        nn.init.zeros_(self.fuse.weight)
        nn.init.zeros_(self.fuse.bias)

    def forward(self, x):
        ca_feat = self.ca(x)
        aspp_feat = self.aspp(x)

        residual = self.fuse(torch.cat([ca_feat, aspp_feat], dim=1))
        return x + residual


class EdgeGuidedResidualGating(nn.Module):
    def __init__(self, init_alpha=0.0, canny_low=50, canny_high=150):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))
        self.canny_low = int(canny_low)
        self.canny_high = int(canny_high)

    def _canny_edge(self, image, out_size):
        b, c, h, w = image.shape
        edge_list = []

        image_cpu = image.detach().float().cpu()

        for i in range(b):
            img = image_cpu[i]

            # [C, H, W] -> grayscale [H, W]
            if img.shape[0] == 3:
                gray = img.mean(dim=0).numpy()
            else:
                gray = img[0].numpy()

            # Normalize to uint8 for OpenCV Canny.
            gray_min = gray.min()
            gray_max = gray.max()
            gray = (gray - gray_min) / (gray_max - gray_min + 1e-6)
            gray = (gray * 255.0).astype(np.uint8)

            # Slight smoothing helps reduce noisy edges.
            gray = cv2.GaussianBlur(gray, (3, 3), 0)

            edge = cv2.Canny(gray, self.canny_low, self.canny_high)
            edge = edge.astype(np.float32) / 255.0

            edge_tensor = torch.from_numpy(edge).unsqueeze(0).unsqueeze(0)
            edge_list.append(edge_tensor)

        edge = torch.cat(edge_list, dim=0)
        edge = edge.to(device=image.device, dtype=image.dtype)

        edge = F.interpolate(
            edge,
            size=out_size,
            mode='bilinear',
            align_corners=False
        )

        return edge

    def forward(self, x, image=None, edge=None):
        if edge is None:
            if image is None:
                return x
            edge = self._canny_edge(image, x.shape[-2:])
        else:
            if edge.dim() == 3:
                edge = edge.unsqueeze(1)

            edge = edge.to(device=x.device, dtype=x.dtype)

            edge = F.interpolate(
                edge,
                size=x.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

        return x * (1.0 + self.alpha * edge)

class SACR(nn.Module):
    def __init__(self, cfg, clip_model, rank, zeroshot_weights=None):
        super().__init__()

        self.vit = VisionTransformer(
            clip_model=clip_model,
            input_resolution=224,
            patch_size=16,
            width=768,
            layers=12,
            heads=12,
            output_dim=768
        )

        self.clip = clip_model
        self.k = getattr(cfg.DATASET, "K", 0)

        visual_channel = cfg.MODEL.VISUAL_CHANNEL
        text_channel = cfg.MODEL.TEXT_CHANNEL

        self.proj = nn.Conv2d(visual_channel, text_channel, 1, bias=False)
        self._initialize_weights(clip_model)

        self.logit_scale = clip_model.logit_scale

        for p in self.parameters():
            p.requires_grad = False

        self.text_encoder = TextEncoder(
            clip_model,
            training=cfg.MODEL.TRAINING,
            cfg=cfg,
            device=rank
        )

        self.cnum = cfg.DATASET.NUM_CLASSES
        self.device = rank

        self.pe_proj = nn.Conv2d(768, 512, kernel_size=1)
        self.decoder_conv2 = nn.Conv2d(self.cnum + 512, self.cnum, kernel_size=5, padding=2, stride=1)
        self.decoder_norm2 = nn.BatchNorm2d(self.cnum)

        if cfg.MODEL.TRAINING:
            nn.init.kaiming_normal_(self.decoder_conv2.weight, a=0, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(self.decoder_norm2.weight, 1)
            nn.init.constant_(self.decoder_norm2.bias, 0)

        self.use_context_refine = bool(getattr(cfg.MODEL, "USE_CONTEXT_REFINE", False))
        self.train_context_refine = bool(getattr(cfg.MODEL, "TRAIN_CONTEXT_REFINE", False))

        self.use_edge_gate = bool(getattr(cfg.MODEL, "USE_EDGE_GATE", False))
        self.train_edge_gate = bool(getattr(cfg.MODEL, "TRAIN_EDGE_GATE", False))

        self.bias_lambda = float(getattr(cfg.MODEL, "BIAS_LAMBDA", 1.0))
        self.use_relu_rect = bool(getattr(cfg.MODEL, "USE_RELU_RECT", False))

        print(
            "[BR-SACR CFG] "
            f"use_context_refine={self.use_context_refine}, "
            f"train_context_refine={self.train_context_refine}, "
            f"use_edge_gate={self.use_edge_gate}, "
            f"train_edge_gate={self.train_edge_gate}, "
            f"edge_alpha={float(getattr(cfg.MODEL, 'EDGE_ALPHA', 0.0))}, "
            f"use_relu_rect={self.use_relu_rect}"
        )
        self.context_refine = StructureAwareContextRefinement(
            channels=self.cnum,
            reduction=int(getattr(cfg.MODEL, "CA_REDUCTION", 32)),
            dilations=tuple(getattr(cfg.MODEL, "ASPP_DILATIONS", [1, 2, 3]))
        )

        self.edge_gate = EdgeGuidedResidualGating(
            init_alpha=float(getattr(cfg.MODEL, "EDGE_ALPHA", 0.0))
        )

        for p in self.parameters():
            p.requires_grad = False

        self.text_encoder.prompt_token.requires_grad = True

        for p in self.pe_proj.parameters():
            p.requires_grad = True

        for p in self.decoder_conv2.parameters():
            p.requires_grad = True

        for p in self.decoder_norm2.parameters():
            p.requires_grad = True

        for p in self.context_refine.parameters():
            p.requires_grad = False

        for p in self.edge_gate.parameters():
            p.requires_grad = False

        train_context_refine = bool(getattr(cfg.MODEL, "TRAIN_CONTEXT_REFINE", False))
        train_edge_gate = bool(getattr(cfg.MODEL, "TRAIN_EDGE_GATE", False))

        if train_context_refine:
            for p in self.context_refine.parameters():
                p.requires_grad = True

        if train_edge_gate:
            for p in self.edge_gate.parameters():
                p.requires_grad = True

    def forward(self, image, gt_cls, zeroshot_weights, cls_name_token,
                training=False, img_metas=None, return_feat=False, edge=None, **kwargs):

        cnum = zeroshot_weights.shape[0]
        device = self.device
        gt_cls_text_embeddings = zeroshot_weights.to(device)

        batch_size = image.shape[0]
        image = image.to(device)

        v, shape, z_global, k, positional_embedding = self.vit(
            image,
            train=False,
            img_metas=img_metas
        )

        positional_embedding = positional_embedding.reshape(
            1, shape[0], shape[1], -1
        ).permute(0, 3, 1, 2)

        feat = self.proj(v)
        feat = feat / (feat.norm(dim=1, keepdim=True) + 1e-6)

        logit_scale = self.logit_scale.exp()

        output_q = F.conv2d(
            feat,
            gt_cls_text_embeddings[:, :, None, None]
        ).permute(0, 2, 3, 1).reshape(batch_size, -1, cnum)

        prompt = self.text_encoder(cls_name_token)
        prompt = prompt / (prompt.norm(dim=-1, keepdim=True) + 1e-6)

        pe = self.pe_proj(positional_embedding).permute(0, 2, 3, 1).reshape(
            1, shape[0] * shape[1], -1
        )

        bias_logits = pe @ prompt.t()

        output = torch.sub(output_q, self.bias_lambda * bias_logits).permute(0, 2, 1).reshape(
            batch_size, -1, shape[0], shape[1]
        )

        if self.use_relu_rect:
            output = F.relu(output)

        feature = torch.cat((feat, output), dim=1)
        output = self.decoder_conv2(feature)
        output = self.decoder_norm2(output)

        if self.use_context_refine:
            output = self.context_refine(output)

        if self.use_edge_gate:
            output = self.edge_gate(output, image=image, edge=edge)

        if return_feat:
            return output[0], feat[0], shape

        if training:
            output_scale = torch.mul(
                output.reshape(batch_size, cnum, -1).permute(0, 2, 1),
                100
            )

            output_gumbel = F.gumbel_softmax(
                output_scale,
                tau=1,
                hard=True,
                dim=2
            ).reshape(batch_size, shape[0], shape[1], -1)

            loss = 0

            for j in range(batch_size):
                masked_image_features = []

                if len(gt_cls[j]) == 0:
                    continue

                for i in gt_cls[j]:
                    i = int(i)

                    if i < 0 or i >= cnum:
                        continue

                    mask = output_gumbel[j, :, :, i].unsqueeze(dim=0)
                    masked_image_feature = torch.mul(feat[j].unsqueeze(dim=0), mask)
                    feature_pool = nn.AdaptiveAvgPool2d((1, 1))(masked_image_feature).reshape(1, 512)
                    masked_image_features.append(feature_pool)

                if len(masked_image_features) == 0:
                    continue

                masked_image_features = torch.stack(masked_image_features, dim=0).squeeze(dim=1)

                similarity_img = logit_scale * masked_image_features @ gt_cls_text_embeddings.t()
                labels = torch.tensor(gt_cls[j], dtype=torch.long, device=device)

                loss += F.cross_entropy(similarity_img, labels)

            return output, loss / batch_size

        return output

    def _initialize_weights(self, clip_model):
        self.proj.weight = nn.Parameter(
            clip_model.visual.proj[:, :, None, None].permute(1, 0, 2, 3).to(torch.float32),
            requires_grad=False
        )
