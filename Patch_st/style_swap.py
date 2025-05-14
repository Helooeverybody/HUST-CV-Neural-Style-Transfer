# style_swap.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def extract_patches(feature_map, patch_size=3, stride=1):
    """Extracts patches from a feature map.

    Args:
        feature_map (torch.Tensor): Input feature map (B x C x H x W).
        patch_size (int): Size of the square patches.
        stride (int): Stride for patch extraction.

    Returns:
        torch.Tensor: Extracted patches (B * n_patches_h * n_patches_w, C, patch_size, patch_size).
    """
    B, C, H, W = feature_map.shape
    # Use unfold to extract patches
    # unfold(dimension, size, step)
    patches = F.unfold(feature_map, kernel_size=patch_size, stride=stride)
    # patches shape: (B, C * patch_size * patch_size, n_patches_h * n_patches_w)
    patches = patches.permute(0, 2, 1).contiguous()
    # patches shape: (B, n_patches_h * n_patches_w, C * patch_size * patch_size)
    n_patches_total = patches.shape[1]
    patches = patches.view(B * n_patches_total, C, patch_size, patch_size)
    return patches


def style_swap_op(
    content_features, style_features, patch_size=3, stride=1, eps=1e-8, verbose=False
):
    """Performs the Style Swap operation.

    Args:
        content_features (torch.Tensor): Content feature map (B x C x H x W).
        style_features (torch.Tensor): Style feature map (B x C x H' x W').
        patch_size (int): Size of the patches.
        stride (int): Stride for patch matching and reconstruction.
        eps (float): Epsilon for numerical stability (e.g., in normalization).

    Returns:
        torch.Tensor: The resulting feature map after style swap (B x C x H x W).
    """
    device = content_features.device
    B_c, C_c, H_c, W_c = content_features.shape
    B_s, C_s, H_s, W_s = style_features.shape

    assert (
        C_c == C_s
    ), "Content and Style features must have the same number of channels."
    C = C_c

    # 1. Extract patches from style features
    # style_patches shape: (B_s * N_s, C, patch_size, patch_size) where N_s is num style patches
    if verbose:
        print(f"Extracting patches from style features...")
        print(f"Style features shape: {style_features.shape}")
        print(f"Content features: \n{content_features}")
        print(f"Style features shape: {style_features.shape}")
        print(f"Style features: \n{style_features}")

    style_patches = extract_patches(style_features, patch_size, stride)
    if verbose:
        print(f"Extracted style patches shape: {style_patches.shape}")
        print(f"Style patches: \n{style_patches}")

    # 2. Normalize style patches (for correlation calculation)
    # norm shape: (B_s * N_s, 1, 1, 1)
    style_patches_norm = torch.sqrt(
        torch.sum(style_patches**2, dim=(1, 2, 3), keepdim=True)
    )
    # Avoid division by zero for zero-patches
    style_patches_normalized = style_patches / (style_patches_norm + eps)

    # 3. Compute correlation using convolution
    # Use style patches as convolution filters
    # conv_filters shape: (n_style_patches, C, patch_size, patch_size)
    conv_filters = style_patches_normalized.to(device)
    # content_features shape: (B_c, C, H_c, W_c)
    # correlation_maps shape: (B_c, n_style_patches, H_out, W_out)
    correlation_maps = F.conv2d(
        content_features, conv_filters, stride=stride, padding=0
    )  # Using 0 padding

    if verbose:
        print(f"Correlation maps shape: {correlation_maps.shape}")
        print(f"Correlation maps: \n{correlation_maps}")

    # 4. Find the best matching style patch for each content patch (Channel-wise Argmax)
    # best_match_indices shape: (B_c, H_out, W_out)
    best_match_indices = torch.argmax(correlation_maps, dim=1)

    if verbose:
        print(f"Best match indices shape: {best_match_indices.shape}")
        print(f"Best match indices: \n{best_match_indices}")

    # 5. Create one-hot selection map
    # one_hot_map shape: (B_c, n_style_patches, H_out, W_out)
    H_out, W_out = best_match_indices.shape[1], best_match_indices.shape[2]
    one_hot_map = torch.zeros_like(correlation_maps, device=device)

    if verbose:
        print(f"One-hot map shape before scatter: {one_hot_map.shape}")
        print(f"One-hot map: \n{one_hot_map}")
    # Use scatter_ to place 1s at the argmax indices
    # scatter_(dimension, index_tensor, value)
    # index needs to be same shape as output after indexing dim -> add channel dim
    one_hot_map.scatter_(1, best_match_indices.unsqueeze(1), 1.0)
    if verbose:
        print(f"One-hot map shape after scatter: {one_hot_map.shape}")
        print(f"One-hot map: \n{one_hot_map}")

    # 6. Reconstruct using transposed convolution
    # Use original (unnormalized) style patches as filters
    # recon_filters shape: (n_style_patches, C, patch_size, patch_size)
    recon_filters = style_patches.to(device)
    # output_padding adjusts output size, often needed if stride > 1
    # calculate required output padding if needed, or ensure input sizes work well
    # For stride=1, output_padding is typically 0
    # swapped_features_sum shape: (B_c, C, H_rec, W_rec) -> should approximate H_c, W_c
    swapped_features_sum = F.conv_transpose2d(
        one_hot_map, recon_filters, stride=stride, padding=0
    )

    if verbose:
        print(f"Swapped features sum shape: {swapped_features_sum.shape}")
        print(f"Swapped features sum: \n{swapped_features_sum}")

    # 7. Normalize for overlapping patches
    # Create filters of ones for counting overlaps
    # count_filters shape: (1, 1, patch_size, patch_size)
    count_filters = torch.ones(1, 1, patch_size, patch_size, device=device)
    # Count contributions per pixel
    # one_hot_map shape: (B_c, n_style_patches, H_out, W_out)
    # Reduce one_hot_map along the patch dimension before counting
    # reduced_one_hot shape: (B_c, 1, H_out, W_out)
    reduced_one_hot = torch.sum(one_hot_map, dim=1, keepdim=True)

    if verbose:
        print(f"Reduced one-hot map shape: {reduced_one_hot.shape}")
        print(f"Reduced one-hot map: \n{reduced_one_hot}")
    # overlap_count shape: (B_c, 1, H_rec, W_rec)
    overlap_count = F.conv_transpose2d(
        reduced_one_hot, count_filters, stride=stride, padding=0
    )

    if verbose:
        print(f"Overlap count shape: {overlap_count.shape}")
        print(f"Overlap count: \n{overlap_count}")

    # Average the contributions
    # Add eps to avoid division by zero where there's no patch contribution
    swapped_features_avg = swapped_features_sum / (overlap_count + eps)

    if verbose:
        print(f"Swapped features average shape: {swapped_features_avg.shape}")
        print(f"Swapped features average: \n{swapped_features_avg}")

    # Ensure output size matches input content feature size if needed (e.g., via padding/cropping)
    # For stride=1 and no padding, output H/W will be slightly smaller than input H/W.
    # If perfect size matching is needed, adjust padding in conv2d/conv_transpose2d
    # or pad/crop the final output. Let's assume approximate size is okay for now.
    # A common strategy is to pad the input `content_features` before conv2d
    # pad_size = patch_size // 2
    # content_features_padded = F.pad(content_features, (pad_size,) * 4, mode='reflect')
    # And then potentially crop the final output to match H_c, W_c

    return swapped_features_avg


if __name__ == "__main__":
    # Example Usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, C, H, W = 1, 3, 5, 5  # Example relu3_1 size for 256x256 input
    content_feat = torch.randn(B, C, H, W).to(device)
    style_feat = torch.randn(B, C, H, W).to(device)  # Assume same size for simplicity

    swapped_feat = style_swap_op(
        content_feat, style_feat, patch_size=3, stride=1, verbose=True
    )
    print(f"Content feature shape: {content_feat.shape}")
    print(f"Style feature shape:   {style_feat.shape}")
    print(f"Swapped feature shape: {swapped_feat.shape}")
