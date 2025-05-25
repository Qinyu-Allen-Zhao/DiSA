import torch
import torch.nn as nn
import math
from timm.models.vision_transformer import DropPath
from timm.layers import SwiGLU
import torch.nn.functional as F
from .rope import *

class RMSNorm(torch.nn.Module):
    def __init__(self, dim, weight=False, eps: float = 1e-6):
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
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        half_head_dim = dim // num_heads // 2
        hw_seq_len = 16
        self.rope = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        #out = flash_attn_qkvpacked_func(qkv).reshape(B, N, C)
        qkv = qkv.permute(2,0,3,1,4)
        q,k,v = qkv.unbind(0)
        q = self.rope(q)
        k = self.rope(k)
        out = F.scaled_dot_product_attention(q,k,v).permute(0,2,1,3).reshape(B,N,C)
        x = self.proj(out)
        x = self.proj_drop(x)
        return x

class CrossAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            semantic_dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim * 1, bias=qkv_bias)
        self.kv = nn.Linear(semantic_dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        half_head_dim = dim // num_heads // 2
        hw_seq_len = 16
        self.rope = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
        )

    def forward(self, q, kv) -> torch.Tensor:
        B, N, C = q.shape
        q = self.q(q).reshape(B, N, self.num_heads, self.head_dim)
        kv = self.kv(kv).reshape(B, N, 2, self.num_heads, self.head_dim)
        q = q.permute(0,2,1,3)
        kv = kv.permute(2,0,3,1,4)
        k,v = kv.unbind(0)
        q = self.rope(q)
        k = self.rope(k)
        out = F.scaled_dot_product_attention(q,k,v).permute(0,2,1,3).reshape(B,N,C)
        x = self.proj(out)
        x = self.proj_drop(x)
        return x

class Block_v2(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            semantic_dim:int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = SwiGLU,
            
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
        )
        self.crossattn = CrossAttention(
            dim,
            semantic_dim=semantic_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2_1 = RMSNorm(dim, True)#norm_layer(dim, elementwise_affine=False)
        self.norm2_2 = RMSNorm(semantic_dim, True)
        self.norm3 = RMSNorm(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio*2/3.),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity() 
        self.ada_lin_attn = nn.Linear(dim, 3*dim)#nn.Sequential(nn.SiLU(inplace=False), )
        self.ada_lin_mlp = nn.Linear(dim, 3*dim)
        self.dim=dim

    def forward(self, x, t, c, condition) -> torch.Tensor:
        B,N,C = condition.shape
        gamma1, scale1, shift1 = self.ada_lin_attn(nn.SiLU()(t)).view(B, 1, 3, self.dim).unbind(2)
        gamma2, scale2, shift2 = self.ada_lin_mlp(nn.SiLU()(c+t)).view(B, N, 3, self.dim).unbind(2)
        x = x + self.drop_path1(self.attn(self.norm1(x).mul(scale1.add(1)).add_(shift1)).mul_(gamma1))
        x = x + self.crossattn(self.norm2_1(x), self.norm2_2(condition))
        x = x + self.drop_path2(self.mlp(self.norm3(x).mul(scale2.add(1)).add_(shift2)).mul_(gamma2))
        return x



class Block_v1(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = SwiGLU,
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
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = RMSNorm(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio*2/3.),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity() 
        self.ada_lin = nn.Linear(dim, 6*dim)#nn.Sequential(nn.SiLU(inplace=False), )
        self.dim=dim

    def forward(self, x: torch.Tensor, t, c, condition) -> torch.Tensor:
        B,N,C = condition.shape
        gamma1, gamma2, scale1, scale2, shift1, shift2 = self.ada_lin(nn.SiLU()(t+c)).view(B, N, 6, self.dim).unbind(2)
        x = x + self.drop_path1(self.attn(self.norm1(x).mul(scale1.add(1)).add_(shift1)).mul_(gamma1))
        x = x + self.drop_path2(self.mlp(self.norm2(x).mul(scale2.add(1)).add_(shift2)).mul_(gamma2))
        return x



def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb



class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(model_channels)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True)
        )
        #self.final =nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class SimpleTransformerAdaLN(nn.Module):
    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        z_channels,
        num_res_blocks,
        cross=False
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks

        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)

        self.input_proj = nn.Linear(in_channels, model_channels)

        res_blocks = []
        if cross:
            for i in range(num_res_blocks):
                res_blocks.append(Block_v2(
                model_channels,model_channels//64, semantic_dim=z_channels
            ))
        else:
            for i in range(num_res_blocks):
                res_blocks.append(Block_v1(
                model_channels,model_channels//64
            ))

        self.res_blocks = nn.ModuleList(res_blocks)
        self.final_layer = FinalLayer(model_channels, out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers

        # Zero-out output layers
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, condition):
        t = (t*1000).long()
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param c: conditioning from AR transformer.
        :return: an [N x C x ...] Tensor of outputs.
        """
        x = self.input_proj(x)
        t = self.time_embed(t).unsqueeze(1)
        c = self.cond_embed(condition)


        for block in self.res_blocks:
            x = block(x, t, c, condition)

        return self.final_layer(x, c+t)
