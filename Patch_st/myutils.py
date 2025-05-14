# utils.py
import torch
from torchvision import transforms
from PIL import Image
import os

# VGG preprocessing values
VGG_MEAN = [0.485, 0.456, 0.406]
VGG_STD = [0.229, 0.224, 0.225]


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image(image_path, img_size=None):
    """Loads an image and optionally resizes it."""
    img = Image.open(image_path).convert("RGB")
    if img_size is not None:
        img = img.resize((img_size, img_size), Image.LANCZOS)
    return img


def preprocess_image(img, device):
    """Preprocesses a PIL image for VGG input."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=VGG_MEAN, std=VGG_STD),
        ]
    )
    return transform(img).unsqueeze(0).to(device)


def postprocess_image(tensor):
    """Postprocesses a tensor back to a PIL image."""
    # Ensure tensor is on CPU
    tensor = tensor.squeeze(0).cpu().detach().clone()
    # Denormalize
    mean = torch.tensor(VGG_MEAN).view(3, 1, 1)
    std = torch.tensor(VGG_STD).view(3, 1, 1)
    tensor = tensor * std + mean
    # Clamp values to [0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    # Convert to PIL Image
    img = transforms.ToPILImage()(tensor)
    return img


def save_image(pil_img, save_path):
    """Saves a PIL image."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pil_img.save(save_path)


def match_color_lab(source_pil, target_pil, eps=1e-7):
    """
    Matches the color distribution of the target_pil image to the source_pil image
    using mean and standard deviation matching in LAB color space.
    Args:
        source_pil (PIL.Image): The image whose color distribution is the reference.
        target_pil (PIL.Image): The image whose color distribution will be modified.
        eps (float): A small epsilon to prevent division by zero.
    Returns:
        PIL.Image: The target image with matched color distribution.
    """
    source_pil = source_pil.convert("RGB")
    target_pil = target_pil.convert("RGB")

    source_np = np.array(source_pil)
    target_np = np.array(target_pil)

    source_lab = sk_color.rgb2lab(source_np)
    target_lab = sk_color.rgb2lab(target_np)

    for i in range(3):  # L, A, B channels
        mu_s = np.mean(source_lab[:, :, i])
        sigma_s = np.std(source_lab[:, :, i])
        mu_t = np.mean(target_lab[:, :, i])
        sigma_t = np.std(target_lab[:, :, i])

        channel_norm = (target_lab[:, :, i] - mu_t) / (sigma_t + eps)
        target_lab[:, :, i] = channel_norm * sigma_s + mu_s

    # skimage's lab2rgb can produce values slightly outside [0,1] due to gamut differences or float precision
    # It also expects L in [0, 100], a, b in approx [-128, 127]
    # We clip after conversion to RGB [0,1] range
    matched_rgb_np = sk_color.lab2rgb(target_lab)

    matched_rgb_np = np.clip(matched_rgb_np, 0, 1)  # Clip to valid [0,1] range for RGB
    matched_rgb_uint8 = (matched_rgb_np * 255).astype(np.uint8)

    return Image.fromarray(matched_rgb_uint8)


def resize_pil_image(pil_img, percentage):
    """Resizes a PIL image to a given percentage of its original size, maintaining aspect ratio."""
    if percentage == 100:
        return pil_img

    scale_factor = percentage / 100.0
    new_width = int(pil_img.width * scale_factor)
    new_height = int(pil_img.height * scale_factor)

    # Ensure dimensions are at least 1x1
    new_width = max(1, new_width)
    new_height = max(1, new_height)

    return pil_img.resize((new_width, new_height), Image.LANCZOS)


if __name__ == "__main__":
    # Example (requires a dummy image file 'dummy.jpg')
    try:
        # Create a dummy image file for testing
        Image.new("RGB", (60, 30), color="red").save("dummy.jpg")

        device = get_device()
        print(f"Using device: {device}")
        pil_img = load_image("dummy.jpg", img_size=256)
        print(f"Loaded image size: {pil_img.size}")
        tensor_img = preprocess_image(pil_img, device)
        print(f"Preprocessed tensor shape: {tensor_img.shape}")
        restored_pil_img = postprocess_image(tensor_img)
        print(f"Restored PIL image size: {restored_pil_img.size}")
        save_image(restored_pil_img, "output/dummy_processed.jpg")
        print("Saved processed dummy image to output/dummy_processed.jpg")
        # Clean up dummy file
        os.remove("dummy.jpg")
    except FileNotFoundError:
        print("Please create a dummy image file 'dummy.jpg' to run the utils example.")
    except Exception as e:
        print(f"An error occurred: {e}")
        if os.path.exists("dummy.jpg"):
            os.remove("dummy.jpg")
