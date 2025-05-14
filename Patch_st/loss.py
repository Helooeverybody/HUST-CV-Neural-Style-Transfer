# loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class TVLoss(nn.Module):
    """Total Variation Loss"""

    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input image tensor (B x C x H x W).
        Returns:
            torch.Tensor: Scalar TV loss value.
        """
        batch_size = x.size(0)
        h_x = x.size(2)
        w_x = x.size(3)
        count_h = (h_x - 1) * w_x
        count_w = h_x * (w_x - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :-1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :-1]), 2).sum()
        loss = self.weight * (h_tv / count_h + w_tv / count_w) / batch_size
        return loss


if __name__ == "__main__":
    # Example Usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_image = torch.randn(4, 3, 256, 256, requires_grad=True).to(device)
    tv_loss = TVLoss(weight=1e-5)
    loss_val = tv_loss(dummy_image)
    print(f"TV Loss: {loss_val.item()}")
    loss_val.backward()
    print(f"Gradient computed: {dummy_image.grad is not None}")
