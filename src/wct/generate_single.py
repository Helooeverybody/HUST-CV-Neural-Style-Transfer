import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import torchvision.transforms.v2 as transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm
import math
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os

from model import MultiLevelAE_OST
import argparse

to_tensor_transforms = transforms.Compose([transforms.ToImage(),transforms.ToDtype(torch.float32,scale=True)])
to_img_transforms = transforms.ToPILImage()

def to_tensor(pil_img):
    img = to_tensor_transforms(pil_img).unsqueeze(0)
    return img

def to_img(tensor: torch.Tensor,content_img):
    tensor = tensor.cpu()
    tensor = tensor.clamp(0, 1).squeeze(0)
    img = to_img_transforms(tensor)
    size = content_img.size
    img = img.resize(size)
    return img 

def img_resize(image, rescale):
    return image.resize((int(image.size[0]*rescale), int(image.size[1]*rescale)))

def color_injection(c_path,output, color_retention_ratio):
    content_img_bgr = cv2.imread(c_path)
    stylized_numpy_array = np.array(output)
    content_img_ycrcb = cv2.cvtColor(content_img_bgr, cv2.COLOR_BGR2YCrCb)
    stylized_img_ycrcb = cv2.cvtColor(stylized_numpy_array, cv2.COLOR_RGB2YCrCb)
    content_y, content_cr, content_cb = cv2.split(content_img_ycrcb)
    stylized_y_from_styled, stylized_cr_from_styled, stylized_cb_from_styled = cv2.split(stylized_img_ycrcb)
    
    final_luminance = stylized_y_from_styled # This is the Y channel from the BGR stylized image

    content_cr_float = content_cr.astype(np.float32)
    content_cb_float = content_cb.astype(np.float32)
    stylized_cr_float = stylized_cr_from_styled.astype(np.float32) # From the (mono)stylized image
    stylized_cb_float = stylized_cb_from_styled.astype(np.float32) # From the (mono)stylized image
    
    blended_cr_float = (color_retention_ratio * content_cr_float +
                        (1 - color_retention_ratio) * stylized_cr_float)
    blended_cb_float = (color_retention_ratio * content_cb_float +
                        (1 - color_retention_ratio) * stylized_cb_float)

    blended_cr = np.clip(blended_cr_float, 0, 255).astype(np.uint8)
    blended_cb = np.clip(blended_cb_float, 0, 255).astype(np.uint8)

    # --- 6. Combine Final Luminance with Blended Chrominance ---
    final_ycrcb_blended = cv2.merge([final_luminance, blended_cr, blended_cb])

    # --- 7. Convert Back to BGR ---
    final_color_blended_bgr = cv2.cvtColor(final_ycrcb_blended, cv2.COLOR_YCrCb2BGR)
    
    final_img_rgb = cv2.cvtColor(final_color_blended_bgr, cv2.COLOR_BGR2RGB)
    
    final_img_pil = Image.fromarray(final_img_rgb)
    return final_img_pil


def main(style_img_dir,content_img_dir,save_dir, model_path, content_size_mult, style_size_mult, alpha, color_ratio, DEVICE):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiLevelAE_OST(pretrained_path_dir=model_path).to(device)
    model.eval()
    
    content_img = Image.open(content_img_dir).convert("RGB")
    style_img = Image.open(style_img_dir).convert("RGB")
    
    content = img_resize(content_img, content_size_mult)
    style = img_resize(style_img, style_size_mult)
    
    with torch.no_grad():
        content = to_tensor(content).to(DEVICE)
        style = to_tensor(style).to(DEVICE)
        output = model(content, style,alpha=alpha)
    output = to_img(output,content_img)
    final_image = color_injection(content_img_dir,output, color_ratio)
    
    save_path = os.path.join(save_dir, "output_new.png")
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    final_image.save(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--style_img_dir", type=str, default="../../datasets/styles/style_1.jpg"
    )
    parser.add_argument(
        "--content_img_dir", type=str, default = "../../datasets/contents/content_1.jpg"
    )
    parser.add_argument(
        "--save_dir", type=str, default="./output"
    )
    parser.add_argument(
        "--model_path", type=str, default="../../models/wct"
    )
    parser.add_argument(
        "--content_size_mult", type=int, default=0.5
    )
    parser.add_argument(
        "--style_size_mult", type=int, default=0.5
    )
    parser.add_argument(
        "--alpha", type=int, default=1
    )
    parser.add_argument(
        "--color_ratio", type=int, default=1
    )
    parser.add_argument(
        "--device", type=str, default="cuda"
    )
    
    main(
        style_img_dir=parser.parse_args().style_img_dir,
        content_img_dir=parser.parse_args().content_img_dir,
        save_dir=parser.parse_args().save_dir,
        model_path=parser.parse_args().model_path,
        content_size_mult=parser.parse_args().content_size_mult,
        style_size_mult=parser.parse_args().style_size_mult,
        alpha=parser.parse_args().alpha,
        color_ratio=parser.parse_args().color_ratio,
        DEVICE=parser.parse_args().device,
    )
    
    
