import torch
import torchvision.transforms.v2 as transforms
from PIL import Image
import os
from src.wct.model import MultiLevelAE_OST
import argparse
from src.wct.utils import to_img, to_tensor,img_resize, color_injection
from src.patch_st.inverse_net import InverseNetwork
from src.patch_st.vgg import VGG19_LAYER_MAP, Vgg19FeatureExtractor
from src.patch_st.style_swap import style_swap_op
import torch
from PIL import Image
from torchvision import transforms
from src.patch_st.myutils import (
    preprocess_image,
    resize_pil_image,
    match_color_lab,
)
import matplotlib.pyplot as plt
import os
from PIL import Image
content_dir = "data/contents"
style_dir = "data/styles"


def gen_wct(content_img_dir = "data/contents",style_img_dir = "data/styles",
            model_path = "models/wct", content_size_mult = 0.5, 
            style_size_mult = 0.5, alpha = 1, color_ratio = 1,
            save_dir = "results/wct",
            device = "cuda"):
    device = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
    model = MultiLevelAE_OST(pretrained_path_dir=model_path).to(device)
    model.eval()
    
    pair_list = [(content,style) for content in os.listdir(content_img_dir) for style in os.listdir(style_img_dir)]
    for x,hehe in enumerate(pair_list):
        content_path,style_path = hehe
        content_img = Image.open(os.path.join(content_img_dir,content_path)).convert("RGB")
        style_img = Image.open(os.path.join(style_img_dir,style_path)).convert("RGB")
        
        content = img_resize(content_img, content_size_mult)
        style = img_resize(style_img, style_size_mult)
        
        with torch.no_grad():
            content = to_tensor(content).to(device)
            style = to_tensor(style).to(device)
            output = model(content, style,alpha=alpha)
        output = to_img(output,content_img)
        final_image = color_injection(os.path.join(content_img_dir,content_path),output, color_ratio)
        
        save_path = os.path.join(save_dir, f"{os.path.splitext(content_path)[0]}_{os.path.splitext(style_path)[0]}.png")
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        final_image.save(save_path)


def gen_patch_st(content_img_dir = "data/contents",style_img_dir = "data/styles",
                 model_path = "models/patch_st/inverse_net.pth", 
                 resize_percent_content = 50, 
                 resize_percent_style = 50, match_color = True,
                 save_dir = "results/patch_st",
                 device = "cuda"):
    

    device = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
    pair_list = [(content,style) for content in os.listdir(content_img_dir) for style in os.listdir(style_img_dir)]
    vgg_extractor = (
            Vgg19FeatureExtractor(layers_to_extract=["relu3_1"]).to(device).eval()
        )
    
    inverse_net = InverseNetwork(input_channels=256).to(device).eval()
    checkpoint = torch.load(model_path, map_location=device)
    inverse_net.load_state_dict(checkpoint)
    to_pil = transforms.ToPILImage()
    for x,hehe in enumerate(pair_list):
        content_path,style_path = hehe
        content_img = Image.open(os.path.join(content_img_dir,content_path)).convert("RGB")
        style_img = Image.open(os.path.join(style_img_dir,style_path)).convert("RGB")

        content_img_proc = resize_pil_image(content_img, resize_percent_content)
        style_img_proc = resize_pil_image(style_img, resize_percent_style)


        if match_color:
            style_img_proc = match_color_lab(content_img_proc, style_img_proc)

        content_img_proc = preprocess_image(content_img_proc, device)
        style_img_proc = preprocess_image(style_img_proc, device)

        content_features = vgg_extractor(content_img_proc)
        style_features = vgg_extractor(style_img_proc)

        content_features = content_features[VGG19_LAYER_MAP["relu3_1"]]
        style_features = style_features[VGG19_LAYER_MAP["relu3_1"]]

        swapped_features = style_swap_op(
                content_features,
                style_features,
                patch_size=3,
                stride=1,
            )

        stylized_img = inverse_net(swapped_features)
        style_img_clamped = torch.clamp(stylized_img, 0, 1)
        stylized_img_pil = to_pil(style_img_clamped.squeeze(0).cpu())
        final_img = stylized_img_pil.resize(content_img.size, Image.LANCZOS)
        save_path = os.path.join(save_dir, f"{os.path.splitext(content_path)[0]}_{os.path.splitext(style_path)[0]}.png")
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        final_img.save(save_path)


def display_style_transfer_grid(stylize_func, content_paths, style_paths, img_size=(256, 200), figsize_per_cell=(3, 3)):
    
    n_rows = len(content_paths) + 1
    n_cols = len(style_paths) + 1

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * figsize_per_cell[0], n_rows * figsize_per_cell[1]),
        squeeze=False
    )

    for i in range(n_rows):
        for j in range(n_cols):
            axes[i][j].axis('off')

    for j, s_path in enumerate(style_paths, start=1):
        s_path = os.path.join(style_dir,s_path+".jpg")
        img = Image.open(s_path).convert('RGB').resize(img_size)
        axes[0][j].imshow(img)
        axes[0][j].axis('off')
        style_name = s_path.split('/')[-1]
        axes[0][j].set_title(style_name, fontsize=10, pad=5)

    for i, c_path in enumerate(content_paths, start=1):
        c_path = os.path.join(content_dir,c_path+".jpg")
        img = Image.open(c_path).convert('RGB').resize(img_size)
        axes[i][0].imshow(img)
        axes[i][0].axis('off')
        content_name = c_path.split('/')[-1]
        axes[i][0].set_ylabel(content_name, fontsize=10, rotation=0, labelpad=40, va='center')
        axes[i][0].set_title(content_name, fontsize=10, pad=5)

    for i, c_path in enumerate(content_paths, start=1):
        for j, s_path in enumerate(style_paths, start=1):
            stylized = stylize_func(c_path, s_path)
            if isinstance(stylized, Image.Image):
                img = stylized.resize(img_size)
            else:
                img = Image.fromarray(stylized).resize(img_size)
            axes[i][j].imshow(img)
            axes[i][j].axis('off')

    plt.tight_layout()
    plt.show()
def compare_style_models(
    stylize_funcs,
    model_names,
    content_paths,
    style_paths,
    img_size=(256, 200),
    figsize_per_cell=(3, 3)
):
    assert len(stylize_funcs) == len(model_names), "Each model must have a name."
    assert len(content_paths) == len(style_paths), "Content and style lists must be same length."

    n_pairs = len(content_paths)
    n_models = len(stylize_funcs)
    n_cols = n_models + 2
    n_rows = n_pairs

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * figsize_per_cell[0], n_rows * figsize_per_cell[1]),
        squeeze=False
    )
    for i, (c_path, s_path) in enumerate(zip(content_paths, style_paths)):
        # First col: content
        content_path = os.path.join(content_dir,c_path+".jpg")
        content_img = Image.open(content_path).convert('RGB').resize(img_size)
        axes[i][0].imshow(content_img)
        axes[i][0].axis('off')
        if i == 0:
            axes[i][0].set_title('Content', fontsize=12)
        for j, (func, name) in enumerate(zip(stylize_funcs, model_names), start=1):
            stylized = func(c_path, s_path)
            img = stylized.resize(img_size) if isinstance(stylized, Image.Image) else Image.fromarray(stylized).resize(img_size)
            axes[i][j].imshow(img)
            axes[i][j].axis('off')
            if i == 0:
                axes[i][j].set_title(name, fontsize=12)
        style_path = os.path.join(style_dir,s_path+".jpg")
        style_img = Image.open(style_path).convert('RGB').resize(img_size)
        axes[i][-1].imshow(style_img)
        axes[i][-1].axis('off')
        if i == 0:
            axes[i][-1].set_title('Style', fontsize=12)

    plt.tight_layout()
    plt.show()
    
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type = str, default = "patch_st"
    )
    parser.add_argument(
        "--content_size_mult", type=int, default=0.5
    )
    parser.add_argument(
        "--style_size_mult", type=int, default=0.5
    )
    parser.add_argument(
        "--alpha", type=int, default=1
    )
    parser.add_argument(
        "--color_ratio", type=int, default=1
    )
    parser.add_argument(
        "--resize_percent_content", type=int, default = 50
    )
    parser.add_argument(
        "--resize_percent_style", type=int, default = 50
    )
    parser.add_argument(
        "--match_color", type = bool, default = True
    )

    
    if parser.parse_args().model == "wct":
        #get the styled image
        gen_wct(
            content_size_mult=parser.parse_args().content_size_mult,
            style_size_mult=parser.parse_args().style_size_mult,
            alpha=parser.parse_args().alpha,
            color_ratio=parser.parse_args().color_ratio,
        )

    elif parser.parse_args().model == "patch_st":
        # get the styled image
        gen_patch_st(
            resize_percent_content = parser.parse_args().resize_percent_content, 
            resize_percent_style = parser.parse_args().resize_percent_style, 
            match_color = parser.parse_args().match_color,
        )
        