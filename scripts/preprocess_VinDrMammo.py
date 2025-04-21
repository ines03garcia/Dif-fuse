import os
import imageio
import torch
import numpy as np
import torchvision.transforms.functional as Tf
from tqdm import tqdm
from skimage import img_as_ubyte
from PIL import Image

#input_folder = '/home/csantiago/small_dataset'
input_folder = '/home/csantiago/datasets/Vindir-mammoclip/VinDir_preprocessed_mammoclip/images_png'
#output_folder = '/home/csantiago/small_dataset_256'
output_folder = '/home/csantiago/data'
os.makedirs(output_folder, exist_ok=True)

target_size = 256

for folder in tqdm(os.listdir(input_folder)):
    folder_path = os.path.join(input_folder, folder)
    
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(".png"):
            continue
            
        image_path = os.path.join(input_folder, folder, filename)
        img = Image.open(image_path)

        w, h = img.size
        scale = target_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.BILINEAR)
    
        pad_w = (target_size - new_w) // 2
        pad_h = (target_size - new_h) // 2
        padding = (pad_w, pad_h, target_size - new_w - pad_w, target_size - new_h - pad_h)
        img = Tf.pad(img, padding, fill=0)
    
        output_path = os.path.join(output_folder, filename)
        imageio.imwrite(output_path, img)

