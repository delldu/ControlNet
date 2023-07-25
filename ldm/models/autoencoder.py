import torch
import torch.nn as nn
from ldm.modules.diffusionmodules.model import Decoder

import pdb

class AutoencoderKL(nn.Module): 
    def __init__(self,
                version="v1.5",
                embed_dim = 4,
               ):
        super().__init__()
        ddconfig = {'double_z': True,
            'z_channels': 4,
            'resolution': 256,
            'in_channels': 3,
            'out_ch': 3,
            'ch': 128,
            'ch_mult': [1, 2, 4, 4],
            'num_res_blocks': 2,
            'attn_resolutions': [],
            'dropout': 0.0}

        # self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quant_conv = nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        # self.quant_conv -- Conv2d(8, 8, kernel_size=(1, 1), stride=(1, 1))
        self.post_quant_conv = nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        # self.post_quant_conv -- Conv2d(4, 4, kernel_size=(1, 1), stride=(1, 1))
        self.embed_dim = embed_dim

    def decode(self, z):
        # z.size() -- [1, 4, 80, 64]
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        # dec.size() -- [1, 3, 640, 512]
        return dec

    def forward(self, z):
        return self.decode(z)
