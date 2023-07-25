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
from functools import partial
from ldm.util import count_params
from ldm.modules.diffusionmodules.util import make_beta_schedule

import pdb

class DiffusionWrapper(nn.Module):
    def __init__(self, version="v1.5"):
        super().__init__()
        self.version = version
        from cldm.cldm import ControlledUnetModel
        self.diffusion_model = ControlledUnetModel(version)

class DDPM(nn.Module):
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(self,
                version="v1.5",
                timesteps=1000,
                linear_start=1e-4,
                linear_end=2e-2,
                parameterization="eps",  # all assuming fixed variance schedules
            ):
        super().__init__()
        self.version=version
        assert parameterization in ["eps", "x0", "v"], 'currently only supporting "eps" and "x0" and "v"'
        self.parameterization = parameterization
        self.cond_stage_model = None
        self.model = DiffusionWrapper(version)
        count_params(self.model, verbose=True)

        self.register_schedule(beta_schedule="linear", timesteps=timesteps, linear_start=linear_start, linear_end=linear_end)
        logvar = torch.full(fill_value=0.0, size=(self.num_timesteps,))
        self.register_buffer('logvar', logvar)


    def register_schedule(self, beta_schedule="linear", timesteps=1000, linear_start=1e-4, linear_end=2e-2):
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
    def __init__(self, version="v1.5", scale_factor=0.18215):
        super().__init__(version=version)
        self.version = version
        self.scale_factor = scale_factor
        self.instantiate_first_stage(version)
        self.instantiate_cond_stage(version)

    def instantiate_first_stage(self, version):
        from ldm.models.autoencoder import AutoencoderKL
        model = AutoencoderKL(version)
        self.first_stage_model = model.eval()
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def instantiate_cond_stage(self, version):
        if version == "v1.5":
            from ldm.modules.encoders.modules import FrozenCLIPEmbedder
            model = FrozenCLIPEmbedder()
        else:
            from ldm.modules.encoders.modules import FrozenOpenCLIPEmbedder
            model = FrozenOpenCLIPEmbedder()
        self.cond_stage_model = model.eval()
        for param in self.cond_stage_model.parameters():
            param.requires_grad = False

    def get_learned_conditioning(self, c):
        c = self.cond_stage_model.encode(c)
        return c

    def decode_first_stage(self, z):
        z = 1. / self.scale_factor * z
        return self.first_stage_model.decode(z)

