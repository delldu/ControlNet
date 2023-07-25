import torch
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel
import open_clip

from typing import List
import pdb

# for v1.5
class FrozenCLIPEmbedder(nn.Module): # cond_stage_config
    """Uses the CLIP transformer encoder for text (from huggingface)"""
    LAYERS = [
        "last",
        # "pooled",
        # "hidden"
    ]
    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77, layer="last"):
        # clip-vit-base-patch32
        super().__init__()
        assert layer in self.LAYERS

        self.tokenizer = CLIPTokenizer.from_pretrained(version) # <class 'transformers.models.clip.tokenization_clip.CLIPTokenizer'>
        self.transformer = CLIPTextModel.from_pretrained(version) # <class 'transformers.models.clip.modeling_clip.CLIPTextModel'>
        self.device = device # xxxx8888
        self.max_length = max_length
        self.freeze()

        # xxxx8888 self.transformer.text_model.embeddings -- CLIPTextEmbeddings
        # xxxx8888 self.transformer.text_model.encoder -- CLIPEncoder

    def freeze(self):
        self.transformer = self.transformer.eval()
        #self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text: List[str]):
        # ['bag, best quality, extremely detailed']

        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")
        # batch_encoding.keys() -- dict_keys(['input_ids', 'attention_mask'])

        tokens = batch_encoding["input_ids"].to(self.device) # size() -- [1, 77]
        outputs = self.transformer(input_ids=tokens, output_hidden_states=False)
        #  outputs.keys() -- ['last_hidden_state', 'pooler_output']

        z = outputs.last_hidden_state
        return z # z.size() -- [1, 77, 768]


    def encode(self, text: List[str]):
        return self(text)


# for v2.1
class FrozenOpenCLIPEmbedder(nn.Module):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    LAYERS = [
        #"pooled",
        # "last",
        "penultimate"
    ]
    def __init__(self, arch="ViT-H-14", version="laion2b_s32b_b79k", device="cuda", max_length=77, layer="penultimate"):
        super().__init__()
        assert layer in self.LAYERS

        model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'), pretrained=version)
        del model.visual
        self.model = model
        # (Pdb) self.model
        # CLIP(
        #   (transformer): Transformer(
        #     (resblocks): ModuleList(
        #       (0-23): 24 x ResidualAttentionBlock(
        #         (ln_1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        #         (attn): MultiheadAttention(
        #           (out_proj): NonDynamicallyQuantizableLinear(in_features=1024, out_features=1024, bias=True)
        #         )
        #         (ls_1): Identity()
        #         (ln_2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        #         (mlp): Sequential(
        #           (c_fc): Linear(in_features=1024, out_features=4096, bias=True)
        #           (gelu): GELU(approximate='none')
        #           (c_proj): Linear(in_features=4096, out_features=1024, bias=True)
        #         )
        #         (ls_2): Identity()
        #       )
        #     )
        #   )
        #   (token_embedding): Embedding(49408, 1024)
        #   (ln_final): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        # )

        self.device = device
        self.max_length = max_length
        self.freeze()
        self.layer_idx = 1 # 0 -- for 'last' layer

        # self.model.attn_mask.size() -- [77, 77]

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text: List[str]):
        # ['bag, best quality, extremely detailed']
        tokens = open_clip.tokenize(text) # size() -- [1, 77]
        z = self.encode_with_transformer(tokens.to(self.device))
        return z # z.size() [1, 77, 1024]

    def encode_with_transformer(self, tokens):
        x = self.model.token_embedding(tokens)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding # self.model.positional_embedding.size() -- [77, 1024]

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        return x

    def text_transformer_forward(self, x, attn_mask):
        # len(self.model.transformer.resblocks) -- 24
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            x = r(x, attn_mask=attn_mask)

        return x

    def encode(self, text: str):
        # text -- ['bag, best quality, extremely detailed']
        return self(text)

