from functools import partial

import time
import numpy as np
from tqdm import tqdm
import scipy.stats as stats
import math
import torch
import torch.nn as nn
import torch.nn as nn
from timm.layers import SwiGLU
from timm.models.vision_transformer import DropPath
from typing import Optional
import torch.nn.functional as F
from models.flowmodel import SimpleTransformerAdaLN
from .rope import *
from .flowloss import SILoss
import models.sampler as sampler
import torch.nn as nn
import torch.utils.checkpoint


class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps: float = 1e-6, weight=False):
        super().__init__()
        self.eps = eps
        if weight:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.weight=None

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is None:
            return output
        else:
            return output * self.weight




class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            scale=None
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        half_head_dim = dim // num_heads // 2
        hw_seq_len = 16
        self.rope = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
        )
        self.resolusion = scale
        self.k,self.v=None,None
    def clear_cache(self):
        self.k,self.v=None,None

    def forward(self, x: torch.Tensor, mask) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        sequence = [0,]+[i**2 for i in self.resolusion]
        sequence = torch.cumsum(torch.tensor(sequence),dim=0)
        if self.training:
            q = torch.cat([self.rope(q[:, :, sequence[i]:sequence[i+1]]) for i in range(len(self.resolusion))], dim=2)
            k = torch.cat([self.rope(k[:, :, sequence[i]:sequence[i+1]]) for i in range(len(self.resolusion))], dim=2)
            x = F.scaled_dot_product_attention(
                q, k, v,attn_mask=mask,
                dropout_p=self.attn_drop if self.training else 0.,
            )
        else:
            q= self.rope(q)
            k = self.rope(k)
            if self.k is None or self.v is None:
                self.k = k
                self.v = v
            else:
                self.k = torch.cat([self.k, k], dim=2)
                self.v = torch.cat([self.v, v], dim=2)
            x = F.scaled_dot_product_attention(
                q, self.k, self.v,
            )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = SwiGLU,
            scale=None,
            use_checkpoint=False
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            scale=scale,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = RMSNorm(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio*2/3.),
            act_layer=act_layer,
            drop=proj_drop
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity() 
        self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), nn.Linear(dim, 6*dim))
        self.dim=dim
        self.use_checkpoint=use_checkpoint

    def forward(self, x: torch.Tensor, condition, mask) -> torch.Tensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, condition, mask)
        else:
            return self._forward(x, condition, mask)
    def _forward(self, x: torch.Tensor, condition, mask) -> torch.Tensor:
        gamma1, gamma2, scale1, scale2, shift1, shift2 = self.ada_lin(condition).view(-1, 1, 6, self.dim).unbind(2)
        x = x + self.drop_path1(self.attn(self.norm1(x).mul(scale1.add(1)).add_(shift1), mask).mul_(gamma1))
        x = x + self.drop_path2(self.mlp(self.norm2(x).mul(scale2.add(1)).add_(shift2)).mul_(gamma2))
        return x


class FlowAR(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=256, vae_stride=16, patch_size=1,
                 encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
                 decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 vae_embed_dim=16,
                 label_drop_prob=0.1,
                 class_num=1000,
                 attn_dropout=0.,
                 proj_dropout=0.,
                 buffer_size=0,
                 diffloss_d=3,
                 diffloss_w=1024,
                 scale=(1, 2, 4, 8, 16),
                 cross=False,
                 use_checkpoint=False
                 ):
        super().__init__()

        # --------------------------------------------------------------------------
        # VAE and patchify specifics
        self.vae_embed_dim = vae_embed_dim

        self.img_size = img_size
        self.vae_stride = vae_stride
        self.patch_size = patch_size
        self.seq_h = self.seq_w = img_size // vae_stride // patch_size
        self.scale = list(scale)
        self.seq_len = sum([pz * pz for pz in self.scale])
        self.token_embed_dim = vae_embed_dim * patch_size**2
        

        # --------------------------------------------------------------------------
        # Class Embedding
        self.num_classes = class_num
        self.class_emb = nn.Embedding(1000+1, encoder_embed_dim)
        self.label_drop_prob = label_drop_prob


        # --------------------------------------------------------------------------
        self.z_proj = nn.Linear(self.token_embed_dim, encoder_embed_dim, bias=True)
        
        self.z_proj_ln = RMSNorm(encoder_embed_dim, weight=True)#nn.LayerNorm(encoder_embed_dim, eps=1e-6)
        self.buffer_size = buffer_size
        self.mask_ratio_generator = stats.truncnorm((0.7 - 1.0) / 0.25, 0, loc=1.0, scale=0.25)
        self.encoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len+self.buffer_size, encoder_embed_dim))

        self.encoder_blocks = nn.ModuleList([
            Block(encoder_embed_dim, encoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  proj_drop=proj_dropout, attn_drop=attn_dropout, scale=scale, use_checkpoint=use_checkpoint) for _ in range(encoder_depth)])
        self.encoder_norm =  RMSNorm(encoder_embed_dim, weight=True)

        # --------------------------------------------------------------------------
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.decoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len+self.buffer_size, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  proj_drop=proj_dropout, attn_drop=attn_dropout, scale=scale,use_checkpoint=use_checkpoint) for _ in range(decoder_depth)])

        self.decoder_norm =  RMSNorm(decoder_embed_dim, weight=True) 
        self.diffusion_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len, decoder_embed_dim))

        self.flownet = SimpleTransformerAdaLN(in_channels=self.token_embed_dim,
            model_channels=diffloss_w,
            out_channels=self.token_embed_dim,  # for vlb loss
            z_channels=decoder_embed_dim,
            num_res_blocks=diffloss_d,
            cross=cross)

        self.initialize_weights()

        # --------------------------------------------------------------------------

        
        self.flow_loss_fn = SILoss()

        attention_mask = []
        start=0
        total_length = sum([pz * pz for pz in self.scale])+self.buffer_size
        for idx, pz in enumerate(self.scale):
            pz = pz ** 2
            if idx==0:
                pz+=self.buffer_size
            start += pz
            attention_mask.append(torch.cat([torch.ones((pz, start)),
                                             torch.zeros((pz, total_length - start))], dim=-1))
        # self.variable('constant', 'attention_mask', lambda :jnp.concatenate(attention_mask, axis=0))
        attention_mask = torch.cat(attention_mask, dim=0)
        attention_mask = torch.where(attention_mask == 0, -torch.inf, attention_mask)
        attention_mask = torch.where(attention_mask == 1, 0, attention_mask)
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
        self.register_buffer('mask', attention_mask.contiguous())


    def initialize_weights(self):
        # parameters
        torch.nn.init.normal_(self.class_emb.weight, std=.02)
        torch.nn.init.normal_(self.encoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.diffusion_pos_embed_learned, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def patchify(self, x):
        bsz, c, h, w = x.shape
        p = self.patch_size
        h_, w_ = h // p, w // p

        x = x.reshape(bsz, c, h_, p, w_, p)
        x = torch.einsum('nchpwq->nhwcpq', x)
        x = x.reshape(bsz, h_ * w_, c * p ** 2)
        return x  # [n, l, d]

    def unpatchify(self, x):
        bsz = x.shape[0]
        p = self.patch_size
        c = self.vae_embed_dim
        h_, w_ = self.seq_h, self.seq_w

        x = x.reshape(bsz, h_, w_, c, p, p)
        x = torch.einsum('nhwcpq->nchpwq', x)
        x = x.reshape(bsz, c, h_ * p, w_ * p)
        return x  # [n, c, h, w]


    def forward_mae_encoder(self, x, condition, mask, start=None, end=None):
        if self.training:
            encoder_pos_embed_learned=self.encoder_pos_embed_learned
        else:
            encoder_pos_embed_learned =self.encoder_pos_embed_learned[:, start:end]
        
        # encoder position embedding
        x = x + encoder_pos_embed_learned
        x = self.z_proj_ln(x)

        # apply Transformer blocks
        for blk in self.encoder_blocks:
            x = blk(x, condition, mask)
        x = self.encoder_norm(x)

        return x

    def forward_mae_decoder(self, x, condition, mask, start=None, end=None):
        x = self.decoder_embed(x)
        # decoder position embedding
        if self.training:
            decoder_pos_embed_learned=self.decoder_pos_embed_learned
        else:
            decoder_pos_embed_learned=self.decoder_pos_embed_learned[:, start:end]
        x = x + decoder_pos_embed_learned

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x, condition, mask)
        x = self.decoder_norm(x)[:, self.buffer_size:]
        if self.training:
            diffusion_pos_embed_learned=self.diffusion_pos_embed_learned
        else:
            diffusion_pos_embed_learned=self.diffusion_pos_embed_learned[:, start:end]

        x = x + diffusion_pos_embed_learned
        return x


    def forward(self, imgs, labels):
        
        label_drop = torch.rand(imgs.shape[0],).cuda()<self.label_drop_prob
        fake_label = torch.ones(imgs.shape[0],).cuda()*1000
        labels = torch.where(label_drop, fake_label, labels)
        gt_latents = [imgs.detach()]
        for i in self.scale[::-1][1:]:
            gt_latents.append(F.interpolate(imgs.detach(), (i,i), mode='area'))
        gt_latents=gt_latents[::-1]
        next_scale=self.scale[1:]
        B,C,H,W = imgs.shape
        x_input = [F.interpolate(F.interpolate(gt_latents[idx].detach(), (H,W), mode='bicubic'), (scale,scale), mode='area') for idx,scale in enumerate(next_scale)]

        # class embed
        class_embedding = self.class_emb(labels.long())

        # patchify and mask (drop) tokens
        gt_latents = [self.patchify(i) for i in gt_latents]
        x_input = [self.patchify(i) for i in x_input]
        x_input = torch.cat(x_input, dim=1)
        gt_latents = torch.cat(gt_latents,dim=1)
        x_input = self.z_proj(x_input)
        x_input = torch.cat([class_embedding.unsqueeze(1), x_input], dim=1)
        x_input = torch.cat([class_embedding.unsqueeze(1).repeat(1,self.buffer_size,1), x_input], dim=1)


        # mae encoder
        x = self.forward_mae_encoder(x_input, class_embedding, self.mask)

        # mae decoder
        z = self.forward_mae_decoder(x, class_embedding, self.mask)

        # diffloss
        loss = []
        start = self.buffer_size
        for i in self.scale:
            l = self.flow_loss_fn(self.flownet, gt_latents[:, start:start+i**2], z[:, start:start+i**2]).mean()
            start+=i**2
            loss.append(l/self.scale[-1]**2 * i**2)
        return sum(loss)

    def sample_tokens(self, flow_upper_steps=50, flow_lower_steps=15, guidance=0.9, cfg=1.0, labels=None, progress=False):
        
        if labels is not None:
            class_embedding = self.class_emb(labels)
        else:
            class_embedding = self.class_emb(torch.ones_like(labels).cuda()*1000)
        if not cfg == 1.0:
            class_embedding = torch.cat([class_embedding, self.class_emb(torch.ones_like(labels).cuda()*1000)], dim=0)
 
        x = class_embedding.unsqueeze(1)
        indices = list(range(len(self.scale)))
        if progress:
            indices = tqdm(indices)
        # generate latents
        sequence = [i**2 for i in self.scale]
        sequence = torch.cumsum(torch.tensor(sequence),dim=0)
        starts = torch.cat([torch.tensor([0]), sequence],dim=0)
        total_ar_steps = len(self.scale) - 1
        for blk in self.encoder_blocks:
            blk.attn.clear_cache()
        for blk in self.decoder_blocks:
            blk.attn.clear_cache()
        for step in indices:
            start = starts[step]
            end = sequence[step]
            z = self.forward_mae_encoder(x, class_embedding, None, start, end)
            z = self.forward_mae_decoder(z, class_embedding, None, start, end)
            scaled_cfg = (cfg-1)*step/(len(self.scale)-1)+1
            
            num_step_iter = (flow_upper_steps - flow_lower_steps) * (total_ar_steps - step) / total_ar_steps + flow_lower_steps
            num_step_iter = int(num_step_iter)
            sampled_token_latent = sampler.euler_sampler(self.flownet, torch.randn([z.shape[0], z.shape[1],16]).cuda(), z, num_steps=num_step_iter, cfg_scale=scaled_cfg, guidance_high=guidance).float()
            if not cfg == 1.0:
                z_sample, _ = sampled_token_latent.chunk(2, dim=0)  # Remove null class samples
            else:
                z_sample = sampled_token_latent
            if step==len(self.scale)-1:
                break
            if not cfg == 1.0:
                z_sample = z_sample.repeat(2,1,1)
            x_ = z_sample.detach()
            B,N,C=x_.shape
            x_ = x_.permute(0,2,1).reshape(B,C, self.scale[step], self.scale[step])
            x_= F.interpolate(F.interpolate(x_, (16,16),mode='bicubic'), (self.scale[step+1],self.scale[step+1]), mode='area').reshape(B,C,-1).permute(0,2,1)
            x = self.z_proj(x_)
        tokens = self.unpatchify(z_sample)
        return tokens


def flowar_small(**kwargs):
    model = FlowAR(
        encoder_embed_dim=768, encoder_depth=6, encoder_num_heads=12,
        decoder_embed_dim=768, decoder_depth=6, decoder_num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def flowar_large(**kwargs):
    model = FlowAR(
        encoder_embed_dim=1024, encoder_depth=8, encoder_num_heads=16,
        decoder_embed_dim=1024, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), cross=True, **kwargs)
    return model


def flowar_huge(**kwargs):
    model = FlowAR(
        encoder_embed_dim=1536, encoder_depth=15, encoder_num_heads=16,
        decoder_embed_dim=1536, decoder_depth=15, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), cross=True, use_checkpoint=True, **kwargs)
    return model

