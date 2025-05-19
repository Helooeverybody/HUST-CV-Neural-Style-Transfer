# vgg.py
import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights

VGG19_LAYER_MAP = {
    "relu1_1": 1,
    "relu2_1": 6,
    "relu3_1": 11,  # Layer used
    "relu4_1": 20,
    "relu5_1": 29,
}


class Vgg19FeatureExtractor(nn.Module):
    def __init__(self, layers_to_extract, weights=VGG19_Weights.DEFAULT):
        """
        Initializes the VGG-19 feature extractor.

        Args:
            layers_to_extract (list): List of layer names (e.g., ['relu3_1'])
                                     from which to extract features.
            weights (VGG19_Weights): Pretrained weights to use.
        """
        super().__init__()
        self.layers_to_extract = sorted(
            [VGG19_LAYER_MAP[name] for name in layers_to_extract]
        )
        self.last_layer_index = self.layers_to_extract[-1]

        vgg = vgg19(weights=weights).features
        self.model = nn.Sequential(*[vgg[i] for i in range(self.last_layer_index + 1)])

        # Freeze VGG parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        Extracts features from the specified layers.

        Args:
            x (torch.Tensor): Input image tensor (B x C x H x W).

        Returns:
            dict: A dictionary where keys are layer indices and values
                  are the corresponding feature maps.
        """
        features = {}
        current_layer_idx = 0
        for i, layer in enumerate(self.model):
            x = layer(x)
            if i in self.layers_to_extract:
                features[i] = x
                current_layer_idx += 1
                if current_layer_idx >= len(self.layers_to_extract):
                    break  # No need to compute further
        return features


if __name__ == "__main__":
    # Example Usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vgg_extractor = Vgg19FeatureExtractor(["relu3_1", "relu4_1"]).to(device).eval()
    dummy_image = torch.randn(1, 3, 256, 256).to(device)
    extracted_features = vgg_extractor(dummy_image)
    print("Extracted features shapes:")
    for layer_idx, feat in extracted_features.items():
        print(f"Layer {layer_idx}: {feat.shape}")
    # Example for single layer extraction (as needed for this paper)
    vgg_relu3_1 = Vgg19FeatureExtractor(["relu3_1"]).to(device).eval()
    feat_3_1 = vgg_relu3_1(dummy_image)[VGG19_LAYER_MAP["relu3_1"]]
    print(f"Relu3_1 shape: {feat_3_1.shape}")
