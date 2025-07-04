from torchvision.utils import save_image
from torchvision import transforms
from argparse import Namespace
from PIL import Image
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
sys.path.extend(['/home/csantiago/Dif-fuse'])
os.makedirs("ddpm_sampling_from_healthy_input", exist_ok=True)
os.makedirs("ddpm_sampling_from_anomalous_input", exist_ok=True)

from guided_diffusion import dist_util
from guided_diffusion.VinDrMammo_dataset import *
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict
)


# Load model and diffusion setup
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
        rescale_timesteps=False, # True for ddim
        batch_size=16,
        gpus=torch.cuda.device_count(),
    )
)

args = Namespace(**args)
model, diffusion = create_model_and_diffusion(
    **args_to_dict(args, model_and_diffusion_defaults().keys())
)

if args.gpus > 1:
    model = torch.nn.DataParallel(model)

device = dist_util.dev()
print("Using device:", device)

model.to(device)  # move to GPU before loading weights

# Load weights from the checkpoint
state_dict = dist_util.load_state_dict("/home/csantiago/models/model007000.pt")

# Load into the model
model.load_state_dict(state_dict)
print("Model loaded!")
model.eval()


# Load dataset
dataset = VinDrMammoDataset(
    dataset_root_folder_filepath='/home/csantiago/data/data-healthy_training_mammograms_to_train_DDPM',
    df_path='/home/csantiago/data/metadata/DDPM/grouped_df_train.csv',
    transform=None,
    only_positive=False, # If True save on anomalous directory
    only_negative=True # If True save on healthy directory
)

loader = DataLoader(
     dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
)


# One batch
x_start, _ = next(iter(loader))
x_start = x_start.to(torch.float32)
x_start = x_start.to(device)
save_image(x_start, "ddpm_sampling_from_healthy_input/x_start.png", nrow=4)
print("✅ Original batch saved as x_start.png")

# Forward Process
timestep = torch.tensor([999], device=device) # configure with how many steps there are
x_999 = diffusion.q_sample(x_start, t=timestep)
save_image(x_999, "ddpm_sampling_from_healthy_input/x_999.png", nrow=4)
print("✅ Noised batch saved as x_999.png")

# Reverse process from x_999 instead of pure noise
from functools import partial

x0_predicted = diffusion.p_sample_loop(
    model,
    shape=x_start.shape,
    noise=x_999,                     # Start from x_t
    clip_denoised=True,
    progress=True,
    device=device,
    model_kwargs={},
)

# Save predictions
save_image(x0_predicted, "ddpm_sampling_from_healthy_input/x0_predicted.png", nrow=4)
print("✅ Saved: original and reconstructed images.")
