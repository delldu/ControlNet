"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
import pdb

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like


class DDIMSampler(object):
    def __init__(self, model):
        super().__init__()
        self.model = model # model -- ControlLDM()
        self.ddpm_num_timesteps = model.num_timesteps # 1000

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        # len(self.ddim_timesteps) -- 20
        # self.ddim_timesteps -- array([  1,  51, 101, 151, 201, 251, 301, 351, 401, 451, 501, 551, 601,
        # 651, 701, 751, 801, 851, 901, 951])

        alphas_cumprod = self.model.alphas_cumprod #
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
            alphacums=alphas_cumprod.cpu(),
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

        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))

    # xxxx1111
    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               eta=0.,
               verbose=True,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None, # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
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

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    log_every_t=log_every_t, # 100
                                                    unconditional_guidance_scale=unconditional_guidance_scale, # 9
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    )
        # samples.size() -- [1, 4, 80, 64]
        return samples, intermediates

    # xxxx1111
    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      timesteps=None, 
                      log_every_t=100,
                      unconditional_guidance_scale=1.,
                      unconditional_conditioning=None, 
                      ):
        device = self.model.betas.device
        b = shape[0] # shape -- (4, 4, 80, 64), 4 samples
        img = torch.randn(shape, device=device)
        timesteps = self.ddim_timesteps

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0] # 20

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
            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, 
                      repeat_noise=False,
                      unconditional_guidance_scale=1.,
                      unconditional_conditioning=None,
                      ):
        b, *_, device = *x.shape, x.device
        # x.size() -- [4, 4, 80, 64]
        # t -- tensor([951, 951, 951, 951], device='cuda:0'), index = 19 ...

        # xxxx1111 !!!
        model_t = self.model.apply_model(x, t, c)
        model_uncond = self.model.apply_model(x, t, unconditional_conditioning)
        e_t = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)

        alphas = self.ddim_alphas
        alphas_prev = self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
        sigmas = self.ddim_sigmas

        # Select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # Current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()

        # Direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) # * temperature, temperature == 1.0
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

        return x_prev, pred_x0

