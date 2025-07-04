import torch
import numpy as np
from torchvision.utils import save_image
import sys
sys.path.extend(['/home/csantiago/Dif-fuse'])
from argparse import Namespace

from guided_diffusion import dist_util
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict
)

# 1. Load model and diffusion setup
args = model_and_diffusion_defaults()
args.update(
    dict(
        image_size=256,
        num_channels=128,
        num_res_blocks=2,
        num_heads=1,
        num_heads_upsample=-1,
        num_head_channels=-1,
        attention_resolutions="32,16,8",
        channel_mult="",
        class_cond=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_fp16=False,
        use_new_attention_order=False,
        learn_sigma=False,
        diffusion_steps=1000,
        noise_schedule="linear",
        rescale_timesteps=False,
    )
)

args = Namespace(**args)


model, diffusion = create_model_and_diffusion(
    **args_to_dict(args, model_and_diffusion_defaults().keys())
)

device = (
    torch.cuda.current_device()
    if torch.cuda.is_available()
    else "cpu"
)
model.to(device)

state_dict = dist_util.load_state_dict("/home/csantiago/models/model007000.pt")

# Strip 'module.'
new_state_dict = {}
for k, v in state_dict.items():
    new_key = k.replace("module.", "") if k.startswith("module.") else k
    new_state_dict[new_key] = v

# Load into the model
model.load_state_dict(new_state_dict)

model.eval()


# 2. Sampling config
batch_size = 16
image_size = 256
channels = 1

# 3. Run sampling loop
sample = diffusion.p_sample_loop(
    model,
    (batch_size, channels, image_size, image_size),
    device=device,
)

# 4. Save image
save_image(sample, "sample_output.png", nrow=4)
print("✅ Sample saved as sample_output.png")