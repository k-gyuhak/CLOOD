""" Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in:

'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale'
    - https://arxiv.org/abs/2010.11929

`How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers`
    - https://arxiv.org/abs/2106.10270

The official jax code is released and available at https://github.com/google-research/vision_transformer

DeiT model defs and weights from https://github.com/facebookresearch/deit,
paper `DeiT: Data-efficient Image Transformers` - https://arxiv.org/abs/2012.12877

Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert

Hacked together by / Copyright 2021 Ross Wightman
"""
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn

from timm.models.layers import PatchEmbed
from timm.models.vision_transformer import VisionTransformer, Block

class mySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

class Adapter(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim, self.out_dim = in_dim, out_dim
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, in_dim)
        self.relu = nn.ReLU()

        self.init_weights()

    def forward(self, x):
        h = self.relu(self.fc1(x))
        h = self.fc2(h)
        return x + h

    def init_weights(self):
        for n, p in self.named_parameters():
            p.data = p * 0 + 1e-24 / p.size(0)

class MyBlock(Block):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, latent=None):
        super().__init__(dim, num_heads, mlp_ratio, qkv_bias, drop, attn_drop,
                 drop_path, act_layer, norm_layer)
        
        self.adapter1 = Adapter(dim, latent)
        self.adapter2 = Adapter(dim, latent)

    def forward(self, x):
        h = self.adapter1(
            self.drop_path(self.attn(self.norm1(x))),
        )
        x = x + h

        h = self.adapter2(
            self.drop_path(self.mlp(self.norm2(x))),
        )
        x = x + h
        return x

class MyVisionTransformer(VisionTransformer):
    """ Vision Transformer with adapter
    
    Most contents are from timm VisionTransformer
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', latent=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
            latent: (int): latent dimension for the adapter module
        """
        super().__init__(
            img_size, patch_size, in_chans, num_classes, embed_dim, depth,
            num_heads, mlp_ratio, qkv_bias, representation_size, distilled,
            drop_rate, attn_drop_rate, drop_path_rate, embed_layer, norm_layer,
            act_layer, weight_init, 
        )
        self.norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = mySequential(*[
            MyBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=self.norm_layer, act_layer=act_layer, latent=latent)
            for i in range(depth)])
        self.norm = self.norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = mySequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            raise NotImplementedError

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_classifier(x)
        return x

    def forward_classifier(self, x):
        if self.head_dist is not None:
            raise NotImplemented
        else:
            x = self.head(x)
        return x

    def adapter_parameters(self):
        return [p for n, p in self.named_parameters() if 'adapter' in n or 'norm' in n or 'head' in n]

    def adapter_named_parameters(self):
        return ((n, p) for n, p in self.named_parameters() if 'adapter' in n or 'norm' in n or 'head' in n)

    def head_named_parameters(self):
        return ((n, p) for n, p in self.named_parameters() if n or 'head' in n)
    
    def adapter_state_dict(self):
        state_dict = {}
        for n, p in self.state_dict().items():
            if 'adapter' in n or 'norm' in n or 'head' in n:
                state_dict[n] = p
        return state_dict
    
    def head_state_dict(self):
        state_dict = {}
        for n, p in self.state_dict().items():
            if 'head' in n:
                state_dict[n] = p
        return state_dict