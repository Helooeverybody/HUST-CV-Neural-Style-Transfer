# inverse_net.py
import torch
import torch.nn as nn


class InverseNetwork(nn.Module):
    def __init__(self, input_channels=256):  # Channels for relu3_1 of VGG19
        """
        Initializes the Inverse Network based on Appendix Table A2.
        Assumes input is from VGG19's relu3_1 (256 channels).
        """
        super().__init__()

        # Input: 1/4 H x 1/4 W x 256 (relu3_1 for 256x256 image)
        self.layers = nn.Sequential(
            # Conv-InstanceNorm-ReLU Block 1
            nn.ConvTranspose2d(input_channels, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),  # Changed to BatchNorm2d for consistency
            nn.PReLU(),
            # Upsampling + Conv-InstanceNorm-ReLU Block 2
            nn.Upsample(scale_factor=2, mode="nearest"),  # To 1/2 H x 1/2 W
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),  # Changed to BatchNorm2d for consistency
            nn.PReLU(),
            # Conv-InstanceNorm-ReLU Block 3
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),  # Changed to BatchNorm2d for consistency
            nn.PReLU(),
            # Upsampling + Conv-InstanceNorm-ReLU Block 4
            nn.Upsample(scale_factor=2, mode="nearest"),  # To H x W
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),  # Changed to BatchNorm2d for consistency
            nn.PReLU(),
            # Output Convolution
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        return self.layers(x)

    def init_weights(self):
        """
        Initialize weights of the network.
        """
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    # Example Usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Example input size for relu3_1 features of a 256x256 image
    dummy_activations = torch.randn(4, 256, 64, 64).to(device)
    inv_net = InverseNetwork(input_channels=256).to(device)
    output_image = inv_net(dummy_activations)
    print(f"Input activation shape: {dummy_activations.shape}")
    print(f"Output image shape: {output_image.shape}")  # Should be (4, 3, 256, 256)
