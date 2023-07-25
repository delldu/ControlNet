from inspect import isfunction
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from typing import Optional, Any
import pdb

try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        self.BxHxNxD_BxNxHD = Rearrange('b h n d -> b n (h d)')
        self.BxNxHxD_BHxNxD = Rearrange('b n h d -> (b h) n d')

    def forward(self, x, context: Optional[torch.Tensor]=None):
        h = self.heads

        q = self.to_q(x)
        # context = default(context, x)
        if context is None:
            context = x

        k = self.to_k(context)
        v = self.to_v(context)

        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        q = self.BxNxHxD_BHxNxD(q.reshape(q.shape[0], q.shape[1], h, -1)) # q.shpe[2] // h
        k = self.BxNxHxD_BHxNxD(k.reshape(k.shape[0], k.shape[1], h, -1))
        v = self.BxNxHxD_BHxNxD(v.reshape(v.shape[0], v.shape[1], h, -1))
        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
        del q, k
    
        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = torch.einsum('b i j, b j d -> b i d', sim, v)
        bh, n, d = out.size()
        # out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        out = self.BxHxNxD_BxNxHD(out.reshape(-1, h, n, d)) # out.shape[0]//h

        return self.to_out(out)


class MemoryEfficientCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        print(f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
              f"{heads} heads.")
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        self.attention_op: Optional[Any] = None

    def forward(self, x, context: Optional[torch.Tensor]=None):
        q = self.to_q(x)
        # context = default(context, x)
        if context is None:
            context = x
        k = self.to_k(context)
        v = self.to_v(context)

        B, HW, C = q.shape # [1, 5120, 320]
        # self.heads -- 8, self.dim_head -- 40
        # q, k, v = map(
        #     lambda t: t.unsqueeze(3)
        #     .reshape(B, t.shape[1], self.heads, self.dim_head)
        #     .permute(0, 2, 1, 3)
        #     .reshape(B * self.heads, t.shape[1], self.dim_head)
        #     .contiguous(),
        #     (q, k, v),
        # )
        q = q.unsqueeze(3).reshape(B, q.shape[1], self.heads, self.dim_head).permute(0, 2, 1, 3) \
            .reshape(B * self.heads, q.shape[1], self.dim_head).contiguous()
        k = k.unsqueeze(3).reshape(B, k.shape[1], self.heads, self.dim_head).permute(0, 2, 1, 3) \
            .reshape(B * self.heads, k.shape[1], self.dim_head).contiguous()
        v = v.unsqueeze(3).reshape(B, v.shape[1], self.heads, self.dim_head).permute(0, 2, 1, 3) \
            .reshape(B * self.heads, v.shape[1], self.dim_head).contiguous()

        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        out = (
            out.unsqueeze(0)
            .reshape(B, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(B, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention
    }
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True):
        super().__init__()
        attn_mode = "softmax-xformers" if XFORMERS_IS_AVAILBLE else "softmax"
        assert attn_mode in self.ATTENTION_MODES
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.attn1 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                              context_dim=None)  # is a self-attention if context is none
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(query_dim=dim, context_dim=context_dim,
                              heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, context: Optional[torch.Tensor]=None):
        x = self.attn1(self.norm1(x), context=None) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x

class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 use_linear=False):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim[d])
                for d in range(depth)]
        )
        if not use_linear:
            self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                                  in_channels,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
        self.use_linear = use_linear # True for v2.1, False for v1.5 !!!

        self.BxCxHxW_BxHWxC = Rearrange('b c h w -> b (h w) c')
        self.BxHxWxC_BxCxHxW = Rearrange('b h w c -> b c h w')

    def forward(self, x, context: Optional[torch.Tensor]=None):
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        # x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        x = self.BxCxHxW_BxHWxC(x).contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks): # len(self.transformer_blocks) -- 1
            x = block(x, context=context)
        if self.use_linear:
            x = self.proj_out(x)
        # x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        x = self.BxHxWxC_BxCxHxW(x.reshape(b, h, w, c)).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in

