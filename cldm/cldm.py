import torch
import torch.nn as nn
import pdb

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, \
    TimestepEmbedSequential, TimestepEmbedSequentialForNormal, \
    ResBlock, Downsample, AttentionBlock
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import exists, instantiate_from_config

import pdb

class ControlledUnetModel(UNetModel):
    '''
        diffusion_model
    '''
    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control:bool=False):
        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)

        if control is not None: # True
            h += control.pop()

        for i, module in enumerate(self.output_blocks): # len(self.output_blocks) -- 12
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)


class ControlNet(nn.Module):
    def __init__(
            self,
            image_size = 32,
            in_channels = 4,
            model_channels = 320,
            hint_channels = 3,
            num_res_blocks = 2,
            attention_resolutions = [4, 2, 1],
            dropout=0.0,
            channel_mult=[1, 2, 4, 4],
            conv_resample=True,
            dims=2,
            use_checkpoint=True,
            use_fp16=False,
            num_heads=8,
            num_head_channels=-1,
            num_heads_upsample=8,
            use_spatial_transformer=True,  # True all for v1.5 and v2.1 custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=768,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=False,
            use_linear_in_transformer=False, # True for v2.1, False for v1.5
    ):
        super().__init__()

        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None: # True
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1: # False
            num_heads_upsample = num_heads

        if num_heads == -1: # False
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1: # False
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_res_blocks = len(channel_mult) * [num_res_blocks] # [2, 2, 2, 2]

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequentialForNormal(conv_nd(dims, in_channels, model_channels, 3, padding=1))
            ]
        )

        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)]) # len(self.zero_convs) -- 12

        self.input_hint_block = TimestepEmbedSequentialForNormal(
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
        )

        self._feature_size = model_channels # 10880
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult): # channel_mult -- [1, 2, 4, 4]
            for nr in range(self.num_res_blocks[level]): # self.num_res_blocks -- [2, 2, 2, 2]
                layers = [
                    ResBlock(ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_scale_shift_norm=False,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions: # [4, 2, 1]
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels

                    layers.append(
                        SpatialTransformer(
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                            use_linear=use_linear_in_transformer,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))

                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequentialForNormal(Downsample(ch, conv_resample, dims=dims, out_channels=out_ch))
                )

                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1: # True
            dim_head = ch // num_heads # 160
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels

        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=False,
            ),
            SpatialTransformer(  # always uses a self-attn
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                use_linear=use_linear_in_transformer,
            ),
            ResBlock(ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=False,
            ),
        )

        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch # ==> 10880


    def make_zero_conv(self, channels):
        return TimestepEmbedSequentialForNormal(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    # def forward(self, x, hint, timesteps, context, **kwargs):
    def forward(self, x, hint, timesteps, context):
        # x.size() -- [1, 4, 80, 64]
        # hint.size() -- [1, 3, 640, 512]
        # timesteps = tensor([951], device='cuda:0')
        # context.size() -- [1, 77, 768]
        # kwargs -- {}

        t_emb = timestep_embedding(timesteps, self.model_channels)
        emb = self.time_embed(t_emb)

        guided_hint = self.input_hint_block(hint, emb, context)

        outs = []

        # xxxx8888
        h = x.type(self.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb, context)
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs # len(outs) -- 13


# xxxx1111, start root ...
class ControlLDM(LatentDiffusion): # DDPM(pl.LightningModule)
    def __init__(self, control_stage_config, control_key, only_mid_control, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13
        # control_key = 'hint'
        # only_mid_control = False
        # args = ()

    def apply_model(self, x_noisy, t, cond):
        # x_noisy.size() -- [1, 4, 80, 64]
        # pp t -- tensor([951], device='cuda:0')
        # cond.keys() -- ['c_concat', 'c_crossattn']
        # args -- (), kwargs -- {}
        assert isinstance(cond, dict)

        diffusion_model = self.model.diffusion_model
        cond_txt = torch.cat(cond['c_crossattn'], 1) # cond['c_crossattn'][0].size() -- [1, 77, 1024]
	    # ==> cond_txt.size() -- [1, 77, 1024]

        # cond['c_concat'][0].size() -- [1, 3, 640, 512]
        if cond['c_concat'] is None: # False
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        else:
            control = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_txt)
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)
        # self.only_mid_control -- False
        # eps.size() -- [1, 4, 80, 64]
        return eps


    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()
