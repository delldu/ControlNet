import torch
from ldm.modules.diffusionmodules.model import Encoder, Decoder
from ldm.util import instantiate_from_config

import pdb

class AutoencoderKL(torch.nn.Module): 
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 image_key="image",
                 monitor='val/rec_loss',
                 ):
        super().__init__()
        # ddconfig = {'double_z': True, 'z_channels': 4, 'resolution': 256, 'in_channels': 3, 'out_ch': 3, 'ch': 128, 'ch_mult': [1, 2, 4, 4], 'num_res_blocks': 2, 'attn_resolutions': [], 'dropout': 0.0}
        # lossconfig = {'target': 'torch.nn.Identity'}
        # embed_dim = 4
        # image_key = 'image'
        # monitor = 'val/rec_loss'

        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)

        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        # self.quant_conv -- Conv2d(8, 8, kernel_size=(1, 1), stride=(1, 1))
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        # self.post_quant_conv -- Conv2d(4, 4, kernel_size=(1, 1), stride=(1, 1))

        self.embed_dim = embed_dim

    # xxxx1111
    def decode(self, z):
        # z.size() -- [1, 4, 80, 64]
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        # dec.size() -- [1, 3, 640, 512]
        return dec

    # xxxx1111, just need decode for inference, so re-define forward for debug    
    # def forward(self, input, sample_posterior=True):
    #     posterior = self.encode(input)
    #     if sample_posterior:
    #         z = posterior.sample()
    #     else:
    #         z = posterior.mode()
    #     dec = self.decode(z)
    #     return dec, posterior

    def forward(self, z):
        return self.decode(z)
