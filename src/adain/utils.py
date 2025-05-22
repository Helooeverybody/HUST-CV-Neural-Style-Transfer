import torch
import numpy as np
import cv2
from PIL import Image
def compute_mean_std(feats:torch.Tensor,eps=1e-8,infer=False)->torch.Tensor:
    assert(len(feats.shape))==4 #N,C,H,W
    if infer:
        n=1
        c=512
    else:
        n,c,_,_=feats.shape
    feats=feats.view([n,c,-1])
    mean=torch.mean(feats,dim=-1).view(n,c,1,1)
    std=torch.std(feats,dim=-1).view(n,c,1,1)+eps

    return mean,std

def inverse_normalize(tensor):
        """Inverse normalize a tensor to original pixel values."""
        mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(tensor.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(tensor.device)
        return tensor * std + mean


def normz(img,mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]):
    if isinstance(img,np.ndarray):
        img=torch.tensor(img)

    mean=torch.tensor(mean).to(img.device)
    std=torch.tensor(std).to(img.device)

    if mean.ndim==1:
        mean=mean.view(-1,1,1)
    if std.ndim==1:
        std=std.view(-1,1,1)

    return (img-mean)/std
def color_injection(content_source, stylized_image,color_retention_ratio=1.0):
    """
    Blend content image colors with stylized image using YCrCb color space.

    Args:
        content_source (str or PIL.Image): Path to the content image or the content image itself.
        stylized_image (PIL.Image): Stylized image in RGB format.

    Returns:
        PIL.Image: Final image with blended colors.
    """
    stylized_numpy_array = np.array(stylized_image)
    output_size = stylized_numpy_array.shape[:2]  # (height, width)

    # Load content image
    if isinstance(content_source, str):
        content_img_bgr = cv2.imread(content_source)
        if content_img_bgr is None:
            raise ValueError(f"Failed to load content image from {content_source}")
    elif isinstance(content_source, Image.Image):
        content_img_bgr = cv2.cvtColor(np.array(content_source), cv2.COLOR_RGB2BGR)
    else:
        raise ValueError("content_source must be a file path (str) or PIL.Image")

    # Resize content image to match stylized image dimensions
    content_img_bgr = cv2.resize(content_img_bgr, (output_size[1], output_size[0]), interpolation=cv2.INTER_AREA)

    # Convert images to YCrCb
    content_img_ycrcb = cv2.cvtColor(content_img_bgr, cv2.COLOR_BGR2YCrCb)
    stylized_img_ycrcb = cv2.cvtColor(stylized_numpy_array, cv2.COLOR_RGB2YCrCb)

    # Split channels
    content_y, content_cr, content_cb = cv2.split(content_img_ycrcb)
    stylized_y, stylized_cr, stylized_cb = cv2.split(stylized_img_ycrcb)

    # Use stylized luminance
    final_luminance = stylized_y

    # Convert chrominance to float for blending
    content_cr_float = content_cr.astype(np.float32)
    content_cb_float = content_cb.astype(np.float32)
    stylized_cr_float = stylized_cr.astype(np.float32)
    stylized_cb_float = stylized_cb.astype(np.float32)

    # Blend chrominance channels
    blended_cr_float = (color_retention_ratio * content_cr_float +
                        (1 - color_retention_ratio) * stylized_cr_float)
    blended_cb_float = (color_retention_ratio * content_cb_float +
                        (1 - color_retention_ratio) * stylized_cb_float)

    # Clip and convert back to uint8
    blended_cr = np.clip(blended_cr_float, 0, 255).astype(np.uint8)
    blended_cb = np.clip(blended_cb_float, 0, 255).astype(np.uint8)

    # Merge channels
    final_ycrcb_blended = cv2.merge([final_luminance, blended_cr, blended_cb])

    # Convert back to BGR, then RGB
    final_color_blended_bgr = cv2.cvtColor(final_ycrcb_blended, cv2.COLOR_YCrCb2BGR)
    final_img_rgb = cv2.cvtColor(final_color_blended_bgr, cv2.COLOR_BGR2RGB)

    return Image.fromarray(final_img_rgb)
def resize_image(image, resize_size=None, resize_ratio=None):
    if not isinstance(image, Image.Image):
        raise ValueError("Input must be a PIL Image object")
    
    try:
        original_width, original_height = image.size
        
        if resize_size is not None:
            # Use resize_size if provided
            target_width = target_height = int(resize_size)
        elif resize_ratio is not None:
            # Calculate dimensions based on resize_ratio
            target_width = int(original_width * resize_ratio)
            target_height = int(original_height * resize_ratio)
        else:
            # Return original image if no resize parameters are provided
            return image
        
        # Ensure dimensions are at least 1 pixel
        target_width = max(1, target_width)
        target_height = max(1, target_height)
        
        # Resize image using PIL's resize method with LANCZOS resampling for better quality
        resized_image = image.resize((target_width, target_height), Image.LANCZOS)
        return resized_image
    except Exception as e:
        raise ValueError(f"Error resizing image: {str(e)}")