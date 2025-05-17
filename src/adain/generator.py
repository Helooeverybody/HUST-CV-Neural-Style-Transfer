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

class Generator:
    """A class for generating stylized images using a style transfer model with color injection."""

    def __init__(self, model, device="auto", color_retention_ratio=1.0):
        """
        Initialize the Generator with a style transfer model and device configuration.

        Args:
            model: The style transfer model (PyTorch model).
            device (str): Device to run the model on ("auto", "cuda", or "cpu").
            color_retention_ratio (float): Ratio for blending content image colors (0 to 1).
        """
        self.model = model
        self.device = self._resolve_device(device)
        self.model = self.model.to(self.device).eval()
        self.color_retention_ratio = color_retention_ratio

    def _resolve_device(self, device):
        """Determine the computation device based on input and availability."""
        device = device.lower()
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA requested but not available. Falling back to CPU.")
            return "cpu"
        print(f"Using device: {device.upper()}")
        return device

    def color_injection(self, content_source, stylized_image):
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
        blended_cr_float = (self.color_retention_ratio * content_cr_float +
                           (1 - self.color_retention_ratio) * stylized_cr_float)
        blended_cb_float = (self.color_retention_ratio * content_cb_float +
                           (1 - self.color_retention_ratio) * stylized_cb_float)

        # Clip and convert back to uint8
        blended_cr = np.clip(blended_cr_float, 0, 255).astype(np.uint8)
        blended_cb = np.clip(blended_cb_float, 0, 255).astype(np.uint8)

        # Merge channels
        final_ycrcb_blended = cv2.merge([final_luminance, blended_cr, blended_cb])

        # Convert back to BGR, then RGB
        final_color_blended_bgr = cv2.cvtColor(final_ycrcb_blended, cv2.COLOR_YCrCb2BGR)
        final_img_rgb = cv2.cvtColor(final_color_blended_bgr, cv2.COLOR_BGR2RGB)

        return Image.fromarray(final_img_rgb)

    def _preprocess_image(self, img_path, resize_size=224):
        """Load and preprocess an image for model input."""
        try:
            img = np.array(Image.open(img_path).convert("RGB"))
            transform = A.Compose([
                A.Resize(resize_size, resize_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
            return transform(image=img)["image"].unsqueeze(0).to(self.device)
        except Exception as e:
            raise ValueError(f"Error processing image {img_path}: {str(e)}")

    def _tensor_to_image(self, tensor):
        """Convert a tensor to a PIL Image without resizing."""
        img = tensor.squeeze().cpu().detach()
        img = self._inverse_normalize(img)
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(img)

    def _postprocess_tensor(self, tensor, output_size):
        """Convert model output tensor to a PIL Image."""
        img = tensor.squeeze().cpu().detach()
        img = self._inverse_normalize(img)
        img = img.permute(1, 2, 0).numpy()
        img = resize(img, output_size, order=1, preserve_range=True)
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(img)

    def _inverse_normalize(self, tensor):
        """Inverse normalize a tensor to original pixel values."""
        mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(tensor.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(tensor.device)
        return tensor * std + mean

    def generate(self, content_dir, style_dir, output_dir,style_size=224, alpha=1.0, apply_color_injection=True):
        """
        Generate stylized images for all content-style pairs from directories.

        Args:
            content_dir (str): Directory containing content images.
            style_dir (str): Directory containing style images.
            output_dir (str): Directory to save stylized images.
            alpha (float): Style strength parameter.
            apply_color_injection (bool): Whether to apply color injection.
        """
        os.makedirs(output_dir, exist_ok=True)
        print("Generating and saving stylized images...")

        # Get lists of image files
        content_images = [f for f in os.listdir(content_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        style_images = [f for f in os.listdir(style_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

        if not content_images or not style_images:
            raise ValueError("No valid images found in content_dir or style_dir")

        for content_file in content_images:
            content_path = os.path.join(content_dir, content_file)
            content_img = Image.open(content_path).convert("RGB")
            content_size = content_img.size[::-1]  # (height, width)
            content_name = os.path.splitext(content_file)[0]

            for style_file in style_images:
                style_path = os.path.join(style_dir, style_file)
                style_name = os.path.splitext(style_file)[0]

                # Preprocess images
                content_tensor = self._preprocess_image(content_path, resize_size=224)
                style_tensor = self._preprocess_image(style_path, resize_size=style_size)

                # Generate stylized image
                with torch.no_grad():
                    stylized_tensor = self.model(content_tensor, style_tensor, alpha=alpha, infer=True)[0]

                # Postprocess
                stylized_img = self._postprocess_tensor(stylized_tensor, content_size)

                # Apply color injection
                if apply_color_injection:
                    stylized_img = self.color_injection(content_img, stylized_img)

                # Save with naming convention
                output_name = f"{content_name}_{style_name}.png"
                output_path = os.path.join(output_dir, output_name)
                stylized_img.save(output_path)

        print("Done generating and saving!")

    def generate_for_a_single_sample(self, content_img_path, style_img_path, alpha=1.0, style_size=224, apply_color_injection=True):
        """
        Generate a stylized image for a single content-style pair, matching content image size.

        Args:
            content_img_path (str): Path to the content image.
            style_img_path (str): Path to the style image.
            alpha (float): Style strength parameter.
            style_size (int): Size to resize style image for preprocessing.
            apply_color_injection (bool): Whether to apply color injection.

        Returns:
            PIL.Image: Stylized image.
        """
        # Load content image to get its size
        content_img = Image.open(content_img_path).convert("RGB")
        output_size = content_img.size[::-1]  # (height, width) for PIL, reversed for numpy/opencv

        content_tensor = self._preprocess_image(content_img_path, resize_size=512)
        style_tensor = self._preprocess_image(style_img_path, resize_size=style_size)

        with torch.no_grad():
            output_tensor = self.model(content_tensor, style_tensor, alpha=alpha, infer=True)[0]

        output_img = self._postprocess_tensor(output_tensor, output_size)

        if apply_color_injection:
            output_img = self.color_injection(content_img, output_img)

        return output_img

