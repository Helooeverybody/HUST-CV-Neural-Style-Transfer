import torch
from PIL import Image
from torchvision import transforms

from Patch_st.myutils import (
    preprocess_image,
    resize_pil_image,
    match_color_lab,
)
from Patch_st.inverse_net import InverseNetwork
from Patch_st.vgg import VGG19_LAYER_MAP, Vgg19FeatureExtractor
from Patch_st.style_swap import style_swap_op


class Inferer:
    """Class to handle the inference process for patch based image style transfer."""

    def __init__(self, checkpoint_path=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vgg_extractor = (
            Vgg19FeatureExtractor(layers_to_extract=["relu3_1"]).to(self.device).eval()
        )
        self.inverse_net = InverseNetwork(input_channels=256).to(self.device).eval()
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.inverse_net.load_state_dict(checkpoint)
        self.to_pil = transforms.ToPILImage()

    def infer(
        self,
        content_img,
        style_img,
        resize_percent_content=50,
        resize_percent_style=50,
        match_color=False,
    ):
        """
        Perform inference to apply style transfer on the content image using the style image.
        Args:
            content_img (PIL.Image): The content image.
            style_img (PIL.Image): The style image.
            resize_percent_content (int): Percentage to resize the content image.
            resize_percent_style (int): Percentage to resize the style image.
            match_color (bool): Whether to match the color of the style image to the content image.
        Returns:
            PIL.Image: The stylized image.
        """
        content_img_proc = resize_pil_image(content_img, resize_percent_content)
        style_img_proc = resize_pil_image(style_img, resize_percent_style)

        if match_color:
            style_img_proc = match_color_lab(content_img_proc, style_img_proc)

        content_img_proc = preprocess_image(content_img_proc, self.device)
        style_img_proc = preprocess_image(style_img_proc, self.device)

        content_features = self.vgg_extractor(content_img_proc)
        style_features = self.vgg_extractor(style_img_proc)

        content_features = content_features[VGG19_LAYER_MAP["relu3_1"]]
        style_features = style_features[VGG19_LAYER_MAP["relu3_1"]]

        swapped_features = style_swap_op(
            content_features,
            style_features,
            patch_size=3,
            stride=1,
        )

        stylized_img = self.inverse_net(swapped_features)

        style_img_clamped = torch.clamp(stylized_img, 0, 1)
        stylized_img_pil = self.to_pil(style_img_clamped.squeeze(0).cpu())

        final_img = stylized_img_pil.resize(content_img.size, Image.LANCZOS)

        return final_img
