import torch
import torch.nn as nn
from pytorch_lightning import seed_everything
import numpy as np

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
    make_ddim_timesteps,
    make_ddim_sampling_parameters,
)

from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, \
    TimestepEmbedSequential, TimestepEmbedSequentialForNormal, \
    ResBlock, Downsample
from ldm.models.diffusion.ddpm import LatentDiffusion

from tqdm import tqdm

from typing import List


import pdb

class ControlledUnetModel(UNetModel):
    '''
        diffusion_model
    '''
    def forward(self, x, timesteps, context, control: List[torch.Tensor]):
        # x.size() -- [1, 4, 80, 64]
        # timesteps -- tensor([801], device='cuda:0')
        # context.size() -- [1, 77, 768]
        # len(control) -- 13, control[0].size() -- [1, 320, 80, 64]

        hs: List[torch.Tensor] = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels)
            emb = self.time_embed(t_emb)
            h = x
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)

        if control is not None: # True
            h += control.pop()

        for i, module in enumerate(self.output_blocks): # len(self.output_blocks) -- 12
            if control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)

        # h = h.type(x.dtype)
        return self.out(h) # self.out -- nn.Sequential from UNetModel

class ControlNet(nn.Module):
    def __init__(
            self,
            version="v1.5",
            in_channels = 4,
            model_channels = 320,
            hint_channels = 3,
            num_res_blocks = 2,
            attention_resolutions = [4, 2, 1],
            dropout=0.0,
            channel_mult=[1, 2, 4, 4],
            conv_resample=True,
            dims=2,
            num_heads_upsample=8,
            transformer_depth=1,  # custom transformer support
    ):
        super().__init__()
        self.version=version
        if version == "v1.5":
            num_heads = 8
            num_head_channels = -1
            context_dim = 768
            use_linear_in_transformer = False
        else:
            # for v2.1 --
            num_heads = -1
            num_head_channels = 64 # need to fix for flash-attn
            context_dim = 1024
            use_linear_in_transformer = True

        if num_heads_upsample == -1: # False
            num_heads_upsample = num_heads

        self.dims = dims
        self.model_channels = model_channels
        self.num_res_blocks = len(channel_mult) * [num_res_blocks] # [2, 2, 2, 2]

        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

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
            ),
            SpatialTransformer(  # always uses a self-attn
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                use_linear=use_linear_in_transformer,
            ),
            ResBlock(ch,
                time_embed_dim,
                dropout,
                dims=dims,
            ),
        )

        self.middle_block_out = self.make_zero_conv(ch)


    def make_zero_conv(self, channels):
        return TimestepEmbedSequentialForNormal(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, hint, timesteps, context):
        # x.size() -- [1, 4, 80, 64]
        # hint.size() -- [1, 3, 640, 512]
        # timesteps = tensor([951], device='cuda:0')
        # context.size() -- [1, 77, 768]

        t_emb = timestep_embedding(timesteps, self.model_channels)
        emb = self.time_embed(t_emb)

        guided_hint = self.input_hint_block(hint, emb, context)

        outs = []

        h = x
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
class ControlLDM(LatentDiffusion): # DDPM(nn.Module)
    def __init__(self, version="v1.5"):
        super().__init__(version=version)
        self.control_model = ControlNet(version)
        self.control_scales = [1.0] * 13

    def apply_model(self, x_noisy, t, cond):
        # x_noisy.size() -- [1, 4, 80, 64]
        # pp t -- tensor([951], device='cuda:0')
        # cond.keys() -- ['c_concat', 'c_crossattn']
        assert isinstance(cond, dict)

        cond_txt = torch.cat(cond['c_crossattn'], 1) # cond['c_crossattn'][0].size() -- [1, 77, 1024]
	    # ==> cond_txt.size() -- [1, 77, 1024]

        # cond['c_concat'][0].size() -- [1, 3, 640, 512]
        if cond['c_concat'] is None: # False
            eps = self.model.diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None)
        else:
            control = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_txt)
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps = self.model.diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control)
        # eps.size() -- [1, 4, 80, 64]
        return eps


    def low_vram_shift(self, is_diffusing: bool):
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


    def forward(self, input_image, a_prompt: str, n_prompt: str, ddim_steps: int, 
        strength: float, scale: float, seed: int, eta: float, save_memory: bool=True):
        B, C, H, W = input_image.size()
        shape = (4, H // 8, W // 8)
        seed_everything(seed)

        if save_memory:
            self.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [input_image], "c_crossattn": [self.get_learned_conditioning([a_prompt] * B)]}
        un_cond = {"c_concat": [input_image], "c_crossattn": [self.get_learned_conditioning([n_prompt] * B)]}

        if save_memory:
            self.low_vram_shift(is_diffusing=True)

        self.control_scales = ([strength] * 13)  # Magic number. 

        samples = self.sample(ddim_steps, B,
                             shape, cond, verbose=False, eta=eta,
                             unconditional_guidance_scale=scale,
                             unconditional_conditioning=un_cond)
        # samples.size() -- [1, 4, 80, 64]

        if save_memory:
            self.low_vram_shift(is_diffusing=False)

        results = self.decode_first_stage(samples)

        return results


    def sample(self,
               S,
               batch_size: int,
               shape: List[int],
               conditioning=None,
               eta: float=0.,
               verbose:bool =True,
               unconditional_guidance_scale: float=1.,
               unconditional_conditioning=None,
               ):

        # S = 20
        # batch_size = 4
        # shape = (4, 80, 64)
        # verbose = False
        # unconditional_guidance_scale = 9

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)

        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta is {eta}') # (1, 4, 96, 64)

        samples = self.ddim_sampling(conditioning, size,
                                    unconditional_guidance_scale=unconditional_guidance_scale, # 9
                                    unconditional_conditioning=unconditional_conditioning,
                                )

        # samples.size() -- [1, 4, 80, 64]
        return samples

    def ddim_sampling(self, cond, shape,
                      unconditional_guidance_scale=1.,
                      unconditional_conditioning=None, 
                      ):

        device = self.betas.device
        b = shape[0] # shape -- (4, 4, 80, 64), 4 samples
        img = torch.randn(shape, device=device)

        time_range = np.flip(self.ddim_timesteps)
        total_steps = self.ddim_timesteps.shape[0] # 20

        print(f"Running DDIM Sampling with {total_steps} timesteps")
        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            outs = self.p_sample_ddim(img, cond, ts, index=index,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,
                                      )
            img, pred_x0 = outs


        return img

    def p_sample_ddim(self, x, c, t, index:int, 
                      unconditional_guidance_scale:float=1.,
                      unconditional_conditioning=None,
                      ):
        b = x.shape[0]
        device = self.betas.device

        # x.size() -- [4, 4, 80, 64]
        # t -- tensor([951, 951, 951, 951], device='cuda:0'), index = 19 ...

        # c.keys() -- dict_keys(['c_concat', 'c_crossattn'])
        # (Pdb) c['c_concat'][0].size() -- [1, 3, 640, 512]
        # (Pdb) c['c_crossattn'][0].size() -- [1, 77, 768]

        # unconditional_conditioning.keys() -- ['c_concat', 'c_crossattn']
        # unconditional_conditioning['c_concat'][0].size() -- [1, 3, 640, 512]
        # unconditional_conditioning['c_crossattn'][0].size() -- [1, 77, 768]

        model_t = self.apply_model(x, t, c)
        model_uncond = self.apply_model(x, t, unconditional_conditioning)
        e_t = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)

        # Select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), self.ddim_alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), self.ddim_alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), self.ddim_sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), self.ddim_sqrt_one_minus_alphas[index], device=device)

        # Current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()

        # Direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * torch.randn(x.shape, device=device) # * temperature, temperature == 1.0
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

        return x_prev, pred_x0


    def make_schedule(self, ddim_num_steps:int, ddim_eta: float=0., verbose: bool=True):
        self.ddim_timesteps = make_ddim_timesteps(num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.num_timesteps,verbose=verbose)
        # len(self.ddim_timesteps) -- 20
        # self.ddim_timesteps -- array([  1,  51, 101, 151, 201, 251, 301, 351, 401, 451, 501, 551, 601,
        # 651, 701, 751, 801, 851, 901, 951])

        assert self.alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
            alphacums=self.alphas_cumprod.cpu(),
            ddim_timesteps=self.ddim_timesteps,
            eta=ddim_eta,
            verbose=verbose)

        # len(ddim_sigmas) -- 20, ddim_sigmas = [0.0, ..., 0.0]

        # (Pdb) len(ddim_alphas) -- 20
        # (Pdb) ddim_alphas
        # tensor([0.9983, 0.9505, 0.8930, 0.8264, 0.7521, 0.6722, 0.5888, 0.5048, 0.4229,
        #         0.3456, 0.2750, 0.2128, 0.1598, 0.1163, 0.0819, 0.0557, 0.0365, 0.0231,
        #         0.0140, 0.0082])

        # (Pdb) len(ddim_alphas_prev) -- 20
        # (Pdb) ddim_alphas_prev
        # array([0.99914998, 0.99829602, 0.95052433, 0.89298052, 0.82639927,
        #        0.75214338, 0.67215145, 0.58881873, 0.50481856, 0.42288151,
        #        0.34555823, 0.27499905, 0.21278252, 0.15981644, 0.11632485,
        #        0.08191671, 0.05571903, 0.03654652, 0.02307699, 0.0140049 ])

        self.ddim_sigmas = ddim_sigmas
        self.ddim_alphas = ddim_alphas
        self.ddim_alphas_prev = ddim_alphas_prev
        self.ddim_sqrt_one_minus_alphas = np.sqrt(1. - ddim_alphas)