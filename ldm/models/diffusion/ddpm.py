"""
wild mixture of
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/CompVis/taming-transformers
-- merci
"""

import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
# from einops import rearrange
# from einops.layers.torch import Rearrange
from functools import partial
from ldm.util import exists, default, count_params, instantiate_from_config
from ldm.modules.diffusionmodules.util import make_beta_schedule

import pdb

# __conditioning_keys__ = {'concat': 'c_concat',
#                          'crossattn': 'c_crossattn',
#                          'adm': 'y'}


class DDPM(pl.LightningModule): # torch.nn.Module, pl.LightningModule
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(self,
                 unet_config,
                 timesteps=1000,
                 # ignore_keys=[],
                 monitor="val/loss",
                 use_ema=True,
                 first_stage_key="image",
                 image_size=256,
                 channels=3,
                 log_every_t=100,
                 linear_start=1e-4,
                 linear_end=2e-2,
                 conditioning_key='crossattn',
                 parameterization="eps",  # all assuming fixed variance schedules
                 ):
        super().__init__()

        assert parameterization in ["eps", "x0", "v"], 'currently only supporting "eps" and "x0" and "v"'
        self.parameterization = parameterization
        # print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
        self.cond_stage_model = None
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.image_size = image_size  # try conv?
        self.channels = channels
        self.model = DiffusionWrapper(unet_config, conditioning_key) # xxxx1111 ????
        count_params(self.model, verbose=True)

        if monitor is not None:
            self.monitor = monitor # 'val/loss_simple_ema'

        self.register_schedule(beta_schedule="linear", timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end)

        logvar = torch.full(fill_value=0.0, size=(self.num_timesteps,))
        self.register_buffer('logvar', logvar)


    def register_schedule(self, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2):
        betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))


class LatentDiffusion(DDPM):
    """main class"""
    def __init__(self,
                 first_stage_config,
                 cond_stage_config,
                 num_timesteps_cond=None,
                 cond_stage_key="image",
                 cond_stage_trainable=False,
                 # concat_mode=True,
                 # cond_stage_forward=None,
                 conditioning_key='crossattn',
                 scale_factor=1.0,
                 scale_by_std=False,
                 *args, **kwargs):
        # num_timesteps_cond = 1
        # cond_stage_key = 'txt'
        # scale_factor = 0.18215
        # args = ()
        # (Pdb) kwargs
        # {'linear_start': 0.00085, 'linear_end': 0.012, 'log_every_t': 200, 'timesteps': 1000, 
        #     'first_stage_key': 'jpg', 'image_size': 64, 'channels': 4, 'monitor': 'val/loss_simple_ema', 
        #     'use_ema': False, 'unet_config': {'target': 'cldm.cldm.ControlledUnetModel', 'params': {'use_checkpoint': True, 'image_size': 32, 'in_channels': 4, 'out_channels': 4, 'model_channels': 320, 'attention_resolutions': [4, 2, 1], 'num_res_blocks': 2, 'channel_mult': [1, 2, 4, 4], 'num_head_channels': 64, 'use_spatial_transformer': True, 'use_linear_in_transformer': True, 'transformer_depth': 1, 'context_dim': 1024, 'legacy': False}}}

        # self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        # assert self.num_timesteps_cond <= kwargs['timesteps']
        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)

        # self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except:
            self.num_downs = 0
        if not scale_by_std: # True
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))
        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)
        # self.cond_stage_forward = cond_stage_forward



    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def instantiate_cond_stage(self, config):
        model = instantiate_from_config(config)
        self.cond_stage_model = model.eval()
        # self.cond_stage_model.train = disabled_train
        for param in self.cond_stage_model.parameters():
            param.requires_grad = False

    # xxxx1111
    def get_learned_conditioning(self, c):
        c = self.cond_stage_model.encode(c)
        return c

    # xxxx1111
    @torch.no_grad()
    def decode_first_stage(self, z):
        z = 1. / self.scale_factor * z
        return self.first_stage_model.decode(z)



# xxxx1111
class DiffusionWrapper(torch.nn.Module): # torch.nn.Module, pl.LightningModule
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.conditioning_key = conditioning_key
        assert self.conditioning_key in [None, 'concat', 'crossattn', 'hybrid', 'adm', 'hybrid-adm', 'crossattn-adm']
