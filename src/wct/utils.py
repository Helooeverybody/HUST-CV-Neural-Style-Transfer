import torch
import torch.nn as nn

import cv2
import torchvision.transforms.v2 as transforms
import numpy as np

from PIL import Image

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