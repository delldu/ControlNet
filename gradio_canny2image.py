from share import *
import config

import cv2
import gradio as gr
import numpy as np
import torch
import random

import einops
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict

import pdb

apply_canny = CannyDetector()
# model = create_model('./models/cldm_v15.yaml').cpu()
model = create_model('v1.5').cpu()

# model = torch.jit.script(model)

# pdb.set_trace()

# torch.jit.script(model.model.diffusion_model)
# torch.jit.script(model.model.diffusion_model.time_embed)
# torch.jit.script(model.model.diffusion_model.input_blocks)
# torch.jit.script(model.model.diffusion_model.middle_block)
# torch.jit.script(model.model.diffusion_model.output_blocks)
# torch.jit.script(model.model.diffusion_model.out)
# pdb.set_trace()

# torch.jit.script(model.first_stage_model) # error
# File "/home/dell/miniconda3/envs/python39/lib/python3.9/site-packages/transformers/models/clip/modeling_clip.py", line 221

# torch.jit.script(model.cond_stage_model.transformer) # CLIPTextModel error
# *** RuntimeError: Can't redefine method: forward on class: __torch__.transformers.models.clip.modeling_clip.CLIPTextEmbeddings (of Python compilation unit at: 0x4fcee80)

# torch.jit.script(model.control_model)

model.load_state_dict(load_state_dict('./models/control_sd15_canny.pth', location='cpu'), strict=False)

# model = torch.compile(model)
model = model.cuda()

# model -- ControlLDM(...),
# model.model -- DiffusionWrapper(...)
# model.first_stage_model -- AutoencoderKL(...)
# model.cond_stage_model -- FrozenCLIPEmbedder(...)
# model.control_model -- ControlNet(...)

def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold):
    img = resize_image(HWC3(input_image), image_resolution) # input_image.shape -- (600, 458, 3), dtype=uint8

    H, W, C = img.shape

    detected_map = apply_canny(img, low_threshold, high_threshold)
    detected_map = HWC3(detected_map) # (640, 512, 3), dtype=uint8

    control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone() # size() -- [1, 3, 640, 512]

    if seed == -1:
        seed = random.randint(0, 65535)

    with torch.no_grad():
        x_samples = model.forward(control, prompt + ', ' + a_prompt, n_prompt, ddim_steps, strength, scale, seed, eta)

    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
    results = [x_samples[i] for i in range(num_samples)]

    return [255 - detected_map] + results


block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## Control Stable Diffusion with Canny Edge Maps")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source='upload', type="numpy")
            prompt = gr.Textbox(label="Prompt")
            run_button = gr.Button(label="Run")
            with gr.Accordion("Advanced options", open=False):
                num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
                strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                guess_mode = gr.Checkbox(label='Guess Mode', value=False)
                low_threshold = gr.Slider(label="Canny low threshold", minimum=1, maximum=255, value=100, step=1)
                high_threshold = gr.Slider(label="Canny high threshold", minimum=1, maximum=255, value=200, step=1)
                ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
                scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                eta = gr.Number(label="eta (DDIM)", value=0.0)
                a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')
                n_prompt = gr.Textbox(label="Negative Prompt",
                                      value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
        with gr.Column():
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
    ips = [input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold]
    run_button.click(fn=process, inputs=ips, outputs=[result_gallery])


block.launch(server_name='0.0.0.0')
