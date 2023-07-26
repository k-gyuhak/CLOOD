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
from copy import deepcopy
import itertools

import torch
import torch.nn as nn

from timm.models.vision_transformer import VisionTransformer, Block
from timm.models.layers import PatchEmbed

class mySequential(nn.Sequential):
    """Deisgned for multiple arguments in nn.Sequential"""
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

class Adapter(nn.Module):
    """Adapter module with HAT
    
    Args:
        in_dim: int, input dimension
        out_dim: int, latent dimension
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim, self.out_dim = in_dim, out_dim
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, in_dim)
        self.relu = nn.ReLU()

        self.gate = torch.sigmoid

        self.ec1 = nn.ParameterList()
        self.ec2 = nn.ParameterList()

        self.init_weights()

    def forward(self, t, x, msk, s):
        masks = self.mask(t, s=s)
        gc1, gc2 = masks

        msk.append(masks)

        h = self.relu(self.mask_out(self.fc1(x), gc1))
        h = self.mask_out(self.fc2(h), gc2)
        return x + h, msk

    def init_weights(self):
        for n, p in self.named_parameters():
            p.data = p * 0 + 1e-24 / p.size(0)

    def mask(self, t, s):
        gc1 = self.gate(s * self.ec1[t])
        gc2 = self.gate(s * self.ec2[t])
        return [gc1, gc2]

    def mask_out(self, out, mask):
        out = out * mask.expand_as(out)
        return out

    def append_embedddings(self):
        self.ec1.append(nn.Parameter(torch.randn(1, self.out_dim, device='cuda')))
        self.ec2.append(nn.Parameter(torch.randn(1, self.in_dim, device='cuda')))

class MyBlock(Block):
    """timm vision_transformer Block with HAT"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, latent=None):
        super().__init__(dim, num_heads, mlp_ratio, qkv_bias, drop, attn_drop,
                 drop_path, act_layer, norm_layer)
        self.list_norm1 = nn.ModuleList()
        self.list_norm2 = nn.ModuleList()

        self.adapter1 = Adapter(dim, latent)
        self.adapter2 = Adapter(dim, latent)

    def forward(self, t, x, msk, s):
        h, msk = self.adapter1(
            t,
            self.drop_path(self.attn(self.list_norm1[t](x))),
            msk,
            s
        )
        x = x + h

        h, msk = self.adapter2(
            t,
            self.drop_path(self.mlp(self.list_norm2[t](x))),
            msk,
            s
        )
        x = x + h
        return t, x, msk, s

class MyVisionTransformer(VisionTransformer):
    """ Vision Transformer with adapter HAT
    
    Most contents are from timm VisionTransformer
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', latent=None, args=None, **kwargs):
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
        super().__init__()
        self.embed_dim = embed_dim

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        self.norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, norm_layer=self.norm_layer) # THIS CONSISTS OF CONV2, NO LAYER_NORM
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = mySequential(*[
            MyBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=self.norm_layer, act_layer=act_layer, latent=latent)
            for i in range(depth)])
        self.norm = self.norm_layer(embed_dim)
        self.list_norm = nn.ModuleList()

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = mySequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        self.head = nn.ModuleList()

    def forward_features(self, t, x, s):
        msk = []
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        _, x, msk, _ = self.blocks(t, x, msk, s)
        x = self.list_norm[t](x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0]), list(itertools.chain(*msk))
        else:
            raise NotImplementedError()

    def forward(self, t, x, s):
        x, msk = self.forward_features(t, x, s=s)
        x = self.forward_classifier(t, x)
        return x, msk

    def forward_classifier(self, t, x):
        if self.head_dist is not None:
            raise NotImplemented
        else:
            x = self.head[t](x)
        return x

    def append_embedddings(self):
        self.head.append(nn.Linear(self.embed_dim, self.num_classes).cuda())

        self.list_norm.append(deepcopy(self.norm))
        for b in self.blocks:
            b.adapter1.append_embedddings()
            b.adapter2.append_embedddings()

            b.list_norm1.append(deepcopy(b.norm1))
            b.list_norm2.append(deepcopy(b.norm2))

    def adapter_parameters(self):
        return [p for n, p in self.named_parameters() if 'adapter' in n or 'list_norm' in n or 'head' in n]

    def head_parameters(self):
        return [p for n, p in self.named_parameters() if 'head' in n]

    def adapter_named_parameters(self):
        return ((n, p) for n, p in self.named_parameters() if 'adapter' in n or 'list_norm' in n or 'head' in n)

    def head_named_parameters(self):
        return ((n, p) for n, p in self.named_parameters() if n or 'head' in n)

    def adapter_state_dict(self):
        state_dict = {}
        for n, p in self.state_dict().items():
            if 'adapter' in n or 'list_norm' in n or 'head' in n:
                state_dict[n] = p
        return state_dict

    def head_state_dict(self):
        state_dict = {}
        for n, p in self.state_dict().items():
            if 'head' in n:
                state_dict[n] = p
        return state_dict
    
    def cum_mask(self, t, p_mask, smax):
        """ 
            Cumulative mask to keep track of all the previous and current mask values. 
            This will be used later as a regularizer in the optimization
        """
        mask = {}
        for n, _ in self.named_parameters():
            names = n.split('.')
            checker = [i for i in ['ec0', 'ec1', 'ec2'] if i in names]
            if names[0] == 'module':
                names = names[1:]
            if checker:
                if 'adapter' in n:
                    gc1, gc2 = self.__getattr__(names[0])[int(names[1])].__getattr__(names[2]).mask(t, s=smax)
                    if checker[0] == 'ec1':
                        n = '.'.join(n.split('.')[:-1]) # e.g. n is layer2.0.ec1.8. The last number 8 represents task id
                        mask[n] = gc1.detach()
                        mask[n].requires_grad = False
                    elif checker[0] == 'ec2':
                        n = '.'.join(n.split('.')[:-1])
                        mask[n] = gc2.detach()
                        mask[n].requires_grad = False

                elif checker[0] == 'ec0': # For ViT, there is no 'ec0', we can discard it
                    n = '.'.join(n.split('.')[:-1])
                    mask[n] = self.mask(t, smax).detach()
                    mask[n].requires_grad = False

        if p_mask is None:
            p_mask = {}
            for n in mask.keys():
                p_mask[n] = mask[n]
        else:
            for n in mask.keys():
                p_mask[n] = torch.max(p_mask[n], mask[n])
        return p_mask

    def freeze_mask(self, p_mask):
        """
            Eq.(2) in the original HAT paper. mask_back is a dictionary whose keys are
            the convolutions' parameter names. Each value of a key is a matrix, whose elements are
            pseudo binary values.

            For ViT Adapter, there's only ec1 and ec2 in adapters in tranformer blocks.
            There are no other ec

            p_mask.keys() are [
            'blocks.0.adapter1.ec1', 'blocks.0.adapter1.ec2',
            'blocks.0.adapter2.ec1', 'blocks.0.adapter2.ec2',
            'blocks.1.adapter1.ec1', 'blocks.1.adapter1.ec2',
            'blocks.1.adapter2.ec1', 'blocks.1.adapter2.ec2',
            'blocks.2.adapter1.ec1', 'blocks.2.adapter1.ec2',
            'blocks.2.adapter2.ec1', 'blocks.2.adapter2.ec2',
            'blocks.3.adapter1.ec1', 'blocks.3.adapter1.ec2',
            'blocks.3.adapter2.ec1', 'blocks.3.adapter2.ec2',
            'blocks.4.adapter1.ec1', 'blocks.4.adapter1.ec2',
            'blocks.4.adapter2.ec1', 'blocks.4.adapter2.ec2',
            'blocks.5.adapter1.ec1', 'blocks.5.adapter1.ec2',
            'blocks.5.adapter2.ec1', 'blocks.5.adapter2.ec2',
            'blocks.6.adapter1.ec1', 'blocks.6.adapter1.ec2',
            'blocks.6.adapter2.ec1', 'blocks.6.adapter2.ec2',
            'blocks.7.adapter1.ec1', 'blocks.7.adapter1.ec2',
            'blocks.7.adapter2.ec1', 'blocks.7.adapter2.ec2',
            'blocks.8.adapter1.ec1', 'blocks.8.adapter1.ec2',
            'blocks.8.adapter2.ec1', 'blocks.8.adapter2.ec2',
            'blocks.9.adapter1.ec1', 'blocks.9.adapter1.ec2',
            'blocks.9.adapter2.ec1', 'blocks.9.adapter2.ec2',
            'blocks.10.adapter1.ec1', 'blocks.10.adapter1.ec2',
            'blocks.10.adapter2.ec1', 'blocks.10.adapter2.ec2',
            'blocks.11.adapter1.ec1', 'blocks.11.adapter1.ec2',
            'blocks.11.adapter2.ec1', 'blocks.11.adapter2.ec2'
            ]
        """
        mask_back = {}
        for n, p in self.named_parameters():
            names = n.split('.')
            if 'adapter' in n: # adapter1 or adapter2. adapter.ec1, adapter.ec2
                # e.g. n is blocks.1.adapter1.fc1.weight
                if 'fc1.weight' in n:
                    mask_back[n] = 1 - p_mask['.'.join(names[:-2]) + '.ec1'].data.view(-1, 1).expand_as(p)
                elif 'fc1.bias' in n:
                    mask_back[n] = 1 - p_mask['.'.join(names[:-2]) + '.ec1'].data.view(-1)
                elif 'fc2.weight' in n:
                    post = p_mask['.'.join(names[:-2]) + '.ec2'].data.view(-1, 1).expand_as(p)
                    pre  = p_mask['.'.join(names[:-2]) + '.ec1'].data.view(1, -1).expand_as(p)
                    mask_back[n] = 1 - torch.min(post, pre)
                elif 'fc2.bias' in n:
                    mask_back[n] = 1 - p_mask['.'.join(names[:-2]) + '.ec2'].view(-1)
        return mask_back