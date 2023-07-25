# pytorch_diffusion + encoder decoder

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from typing import Optional, Any

from ldm.modules.attention import MemoryEfficientCrossAttention
import pdb

try:
    import xformers
    import xformers.ops
    # !!! Set False if you use jit.script mode for torch.jit.script does not support xformers well !!!
    XFORMERS_IS_AVAILBLE = True
    # !!! Set False if you use jit.script mode for torch.jit.script does not support xformers well !!!
except:
    XFORMERS_IS_AVAILBLE = False
    print("No module 'xformers'. Proceeding without it.")

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.temb_proj = nn.Linear(temb_channels, out_channels)

        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else: # To support torch.jit.script
            self.nin_shortcut = nn.Identity()

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x+h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = F.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x + h_

class MemoryEfficientAttnBlock(nn.Module):
    """
        Uses xformers efficient implementation,
        see https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
        Note: this is a single-head self-attention operation
    """
    #
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.attention_op: Optional[Any] = None
        self.BxCxHxW_BxHWxC = Rearrange('b c h w -> b (h w) c')
        self.BxHxWxC_BxCxHxW = Rearrange('b h w c -> b c h w')

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        B, C, H, W = q.shape # (1, 512, 80, 64)
        # q, k, v = map(lambda x: rearrange(x, 'b c h w -> b (h w) c'), (q, k, v))
        q = self.BxCxHxW_BxHWxC(q)
        k = self.BxCxHxW_BxHWxC(k)
        v = self.BxCxHxW_BxHWxC(v)

        # support torch.jit.script
        # q, k, v = map(
        #     lambda t: t.unsqueeze(3) # [1, 5120, 512] --> [1, 5120, 512, 1]
        #     .reshape(B, t.shape[1], 1, C)
        #     .permute(0, 2, 1, 3)
        #     .reshape(B * 1, t.shape[1], C)
        #     .contiguous(),
        #     (q, k, v),
        # )
        q = q.unsqueeze(3).reshape(B, q.shape[1], 1, C).permute(0, 2, 1, 3).reshape(B * 1, q.shape[1], C).contiguous()
        k = k.unsqueeze(3).reshape(B, k.shape[1], 1, C).permute(0, 2, 1, 3).reshape(B * 1, k.shape[1], C).contiguous()
        v = v.unsqueeze(3).reshape(B, v.shape[1], 1, C).permute(0, 2, 1, 3).reshape(B * 1, v.shape[1], C).contiguous()

        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)
        out = out.unsqueeze(0).reshape(B, 1, out.shape[1], C).permute(0, 2, 1, 3).reshape(B, out.shape[1], C)

        # out = rearrange(out, 'b (h w) c -> b c h w', b=B, h=H, w=W, c=C)
        out = self.BxHxWxC_BxCxHxW(out.reshape(B, H, W, C))
        out = self.proj_out(out)
        return x+out


class MemoryEfficientCrossAttentionWrapper(MemoryEfficientCrossAttention):
    def __init__(self):
        suprt(MemoryEfficientCrossAttentionWrapper, self).__init__()
        self.BxCxHxW_BxHWxC = Rearrange('b c h w -> b (h w) c')
        self.BxHxWxC_BxCxHxW = Rearrange('b h w c -> b c h w')

    def forward(self, x, context: Optional[torch.Tensor]=None, mask: Optional[torch.Tensor]=None):
        b, c, h, w = x.shape
        # x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.BxCxHxW_BxHWxC(x)
        out = super().forward(x, context=context, mask=mask) # xxxx8888
        # out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w, c=c)
        out = self.BxHxWxC_BxCxHxW(out.reshape(b, h, w, c))
        return x + out


def make_attn(in_channels, attn_type="vanilla", attn_kwargs=None):
    assert attn_type in ["vanilla", "vanilla-xformers", "memory-efficient-cross-attn", "linear", "none"], f'attn_type {attn_type} unknown'
    if XFORMERS_IS_AVAILBLE and attn_type == "vanilla":
        attn_type = "vanilla-xformers"
    print(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla":
        assert attn_kwargs is None
        return AttnBlock(in_channels)
    elif attn_type == "vanilla-xformers":
        print(f"building MemoryEfficientAttnBlock with {in_channels} in_channels...")
        return MemoryEfficientAttnBlock(in_channels)
    elif type == "memory-efficient-cross-attn":
        attn_kwargs["query_dim"] = in_channels
        return MemoryEfficientCrossAttentionWrapper(**attn_kwargs)
    elif attn_type == "none":
        return nn.Identity(in_channels)
    else:
        raise NotImplementedError()

class Decoder(nn.Module):
    def __init__(self, *, 
                ch=128, 
                out_ch=3, 
                ch_mult=[1,2,4,4], 
                num_res_blocks=2,
                dropout=0.0, 
                in_channels=3,
                resolution=256, 
                z_channels=4,  
                attn_type="vanilla",
                **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        block_in = ch*ch_mult[self.num_resolutions-1]

        # z to block_in
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)): # self.num_resolutions -- 4
            block = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
            up = nn.Module()
            up.block = block
            if i_level != 0:
                up.upsample = Upsample(block_in)
            else: # Support torch.jit.script 
                up.upsample = nn.Identity()

            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def up_layer(self, i:int, h):
        '''Ugly code for torch.jit.script, it does not support reversed(), oh oh oh !!!'''
        for i_level, layer in enumerate(self.up):
            if i_level == i:
                for i_block, block in enumerate(layer.block):
                    h = block(h)
                h = layer.upsample(h)
        return h

    def forward(self, z):
        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # upsampling
        # for i_level in reversed(range(self.num_resolutions)):
        #     for i_block in range(self.num_res_blocks+1):
        #         h = self.up[i_level].block[i_block](h)
        #     h = self.up[i_level].upsample(h)
        for i in range(self.num_resolutions):
            h = self.up_layer(self.num_resolutions - i - 1, h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h
