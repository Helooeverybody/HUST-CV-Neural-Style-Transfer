import torch
import numpy as np
from PIL import Image
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage.transform import resize
import os
import random
import matplotlib.pyplot as plt
from src.adain.utils import resize_image,color_injection,inverse_normalize

class Generator:
    """A class for generating stylized images using a style transfer model with color injection."""

    def __init__(self, model, device, color_retention_ratio=1.0):
        """
        Initialize the Generator with a style transfer model and device configuration.

        Args:
            model: The style transfer model (PyTorch model).
            device (str): Device to run the model on ("auto", "cuda", or "cpu").
            color_retention_ratio (float): Ratio for blending content image colors (0 to 1).
        """
        self.model = model
        self.device = device
        self.model = self.model.to(self.device).eval()
        self.color_retention_ratio = color_retention_ratio

  
    def _preprocess_image(self, img):
        try:
            img=np.array(img)
            transform = A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
            return transform(image=img)["image"].unsqueeze(0).to(self.device)
        except Exception as e:
            raise ValueError(f"Error processing image !")

    def _tensor_to_image(self, tensor):
        """Convert a tensor to a PIL Image without resizing."""
        img = tensor.squeeze().cpu().detach()
        img = inverse_normalize(img)
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(img)

    def _postprocess_tensor(self, tensor, output_size):
        """Convert model output tensor to a PIL Image."""
        img = tensor.squeeze().cpu().detach()
        img = inverse_normalize(img)
        img = img.permute(1, 2, 0).numpy()
        output_size = (output_size[1], output_size[0])
        img = resize(img, output_size, order=1, preserve_range=True)
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(img)

    
    def generate_batch(self, content_dir, style_dir, output_dir,
                 c_size_ratio=0.5,s_size_ratio=0.5,c_size=None,s_size=None, 
                 output_size=None,alpha=1.0, retain_color=True,color_retention_ratio=1.0):
        
        os.makedirs(output_dir, exist_ok=True)
        print("Generating and saving stylized images...")

        # Get lists of image files
        content_images = [f for f in os.listdir(content_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        style_images = [f for f in os.listdir(style_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

        if not content_images or not style_images:
            raise ValueError("No valid images found in content_dir or style_dir")

        for content_file in content_images:
            content_path = os.path.join(content_dir, content_file)
            c_img = Image.open(content_path).convert("RGB")
            c_img=resize_image(c_img,resize_ratio=c_size_ratio,resize_size=c_size)
            content_name = os.path.splitext(content_file)[0]
            
            if not output_size:
                output_size=c_img.size
            for style_file in style_images:
                style_path = os.path.join(style_dir, style_file)
                s_img = Image.open(style_path).convert("RGB")
                s_img=resize_image(s_img,resize_ratio=s_size_ratio,resize_size=s_size)
                style_name = os.path.splitext(style_file)[0]

                # Preprocess images
                content_tensor = self._preprocess_image(c_img)
                style_tensor = self._preprocess_image(s_img)

                # Generate stylized image
                with torch.no_grad():
                    stylized_tensor = self.model(content_tensor, style_tensor, alpha=alpha, infer=True)[0]

                # Postprocess
                stylized_img = self._postprocess_tensor(stylized_tensor, output_size)

                # Apply color injection
                if retain_color:
                    stylized_img = color_injection(c_img, stylized_img,color_retention_ratio)

                # Save with naming convention
                output_name = f"{content_name}_{style_name}.png"
                output_path = os.path.join(output_dir, output_name)
                stylized_img.save(output_path)

        print("Done generating and saving!")

    def generate_single(self, content_path, style_path, alpha=1.0,
                                     c_size_ratio=0.5,s_size_ratio=0.5,c_size=None,s_size=None,
                                    output_size=None,retain_color=True,color_retention_ratio=1.0):
        
        # Load content image to get its size
        c_img = Image.open(content_path).convert("RGB")
        c_img=resize_image(c_img,resize_ratio=c_size_ratio,resize_size=c_size)
        s_img = Image.open(style_path).convert("RGB")
        s_img=resize_image(s_img,resize_ratio=s_size_ratio,resize_size=s_size)
        if not output_size:
            output_size = c_img.size  # (height, width) for PIL, reversed for numpy/opencv

        content_tensor = self._preprocess_image(c_img)
        style_tensor = self._preprocess_image(s_img)

        with torch.no_grad():
            output_tensor = self.model(content_tensor, style_tensor, alpha=alpha, infer=True)[0]

        output_img = self._postprocess_tensor(output_tensor, output_size)

        if retain_color:
            output_img = color_injection(c_img, output_img,color_retention_ratio)

        return output_img