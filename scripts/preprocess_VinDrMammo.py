import os
import imageio
import torch
import numpy as np
from torchvision import transforms
from tqdm import tqdm
from skimage import img_as_ubyte
from PIL import Image

#input_folder = '/home/csantiago/small_dataset'
input_folder = '/home/csantiago/datasets/Vindir-mammoclip/VinDir_preprocessed_mammoclip/images_png'
#output_folder = '/home/csantiago/small_dataset_256'
output_folder = '/home/csantiago/data'
os.makedirs(output_folder, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),  # Converts to [0,1] float32 tensor
])

for folder in tqdm(os.listdir(input_folder)):
    folder_path = os.path.join(input_folder, folder)
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(".png"):
            continue
    
        image_path = os.path.join(input_folder, folder, filename)
        img = Image.open(image_path)
    
        img_tensor = transform(img)
    
        # Convert back to uint8 (.png)
        img_uint8 = img_as_ubyte(img_tensor.squeeze().numpy())
    
        output_path = os.path.join(output_folder, filename)
        imageio.imwrite(output_path, img_uint8)
