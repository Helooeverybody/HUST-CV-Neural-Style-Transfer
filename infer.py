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
from src.rating_model.model import RatingModel
from src.adain.generator import Generator
import numpy as np
from src.adain.model import StyleTransferModel

content_dir = "test_data/contents/"
style_dir = "test_data/styles"
device = "cuda" if torch.cuda.is_available() else "cpu"
def run_wct(content_img_dir,style_img_dir,
            model_path = "models/wct", content_size_mult = 0.5, 
            style_size_mult = 0.5, alpha = 1, color_ratio = 1,
            device = "cuda"):
    device = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
    content_img_dir = content_img_dir + ".jpg"
    style_img_dir = style_img_dir + ".jpg"
    model = MultiLevelAE_OST(pretrained_path_dir=model_path).to(device)
    model.eval()
    

    content_path,style_path = os.path.join(content_dir,content_img_dir), os.path.join(style_dir, style_img_dir)
    content_img = Image.open(content_path).convert("RGB")
    style_img = Image.open(os.path.join(style_path)).convert("RGB")
        
    content = img_resize(content_img, content_size_mult)
    style = img_resize(style_img, style_size_mult)
        
    with torch.no_grad():
        content = to_tensor(content).to(device)
        style = to_tensor(style).to(device)
        output = model(content, style,alpha=alpha)
    output = to_img(output,content_img)
    final_image = color_injection(content_path,output, color_ratio)
    if isinstance(final_image, np.ndarray):
        final_image = Image.fromarray(styled)

        styled = final_image.resize(content_img.size, resample=Image.BICUBIC) 
    return final_image

def run_patch_st(content_img_dir, style_img_dir, model_path = "models/patch_st/inverse_net.pth", 
                 resize_percent_content = 50, 
                 resize_percent_style = 50, match_color = True,
                 device = "cuda"):
    device = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
    content_img_dir = content_img_dir + ".jpg"
    style_img_dir = style_img_dir + ".jpg"

    content_path,style_path = os.path.join(content_dir,content_img_dir), os.path.join(style_dir, style_img_dir)
    content_img = Image.open(content_path).convert("RGB")
    style_img = Image.open(os.path.join(style_path)).convert("RGB")
    vgg_extractor = (
            Vgg19FeatureExtractor(layers_to_extract=["relu3_1"]).to(device).eval()
        )
    
    inverse_net = InverseNetwork(input_channels=256).to(device).eval()
    checkpoint = torch.load(model_path, map_location=device)
    inverse_net.load_state_dict(checkpoint)
    to_pil = transforms.ToPILImage()
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
   
    return final_img


def run_adain(content_img_dir, style_img_dir, model_path = "models/adain/model-ckp176.pth",
              alpha=1.0, style_size=224, apply_color_injection=True,
              device = "cuda"):
    device = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
    content_img_dir = content_img_dir + ".jpg"
    style_img_dir = style_img_dir + ".jpg"

    content_path,style_path = os.path.join(content_dir,content_img_dir), os.path.join(style_dir, style_img_dir)

    device = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
    model = StyleTransferModel(ckp = model_path, device = device)
    gen = Generator(model = model, device = device)
    return gen.generate_for_a_single_sample(content_path, style_path, alpha = alpha,
                                            style_size = style_size, 
                                            apply_color_injection= apply_color_injection)


def predict(image, model_path = 'models/model_rating.pth'):
    model = RatingModel().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    transform = transforms.Compose([
                transforms.Resize([456,456]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalize the image
            ])
    image = transform(image)
    output = model(image.unsqueeze(0).to(device))
    output = output.cpu().detach().numpy()
    print(f"Rating of the image is {output[0][0]*10}")
    return output[0][0] * 10


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type = str, default = "adain"
    )
    parser.add_argument(
        "--content_img_dir", type=str, default = "content_1"
    )
    parser.add_argument(
        "--style_img_dir", type=str, default = "style_1"
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
    parser.add_argument(
        "--alpha_adain", type = int, default = 1.0
    )
    parser.add_argument(
        "--apply_color_injection", type = bool, default = True
    )
    content_img_dir = parser.parse_args().content_img_dir + ".jpg"
    style_img_dir = parser.parse_args().style_img_dir + ".jpg"
    content_path,style_path = os.path.join(content_dir,content_img_dir),\
          os.path.join(style_dir, style_img_dir)
   
    
    if parser.parse_args().model == "wct":
        #get the styled image
        final_img = run_wct(
            style_img_dir=parser.parse_args().style_img_dir,
            content_img_dir=parser.parse_args().content_img_dir,
            content_size_mult=parser.parse_args().content_size_mult,
            style_size_mult=parser.parse_args().style_size_mult,
            alpha=parser.parse_args().alpha,
            color_ratio=parser.parse_args().color_ratio,
        )
       
    elif parser.parse_args().model == "patch_st":
        # get the styled image
        final_img = run_patch_st(
            content_img_dir = parser.parse_args().content_img_dir, 
            style_img_dir = parser.parse_args().style_img_dir,  
            resize_percent_content = parser.parse_args().resize_percent_content, 
            resize_percent_style = parser.parse_args().resize_percent_style, 
            match_color = parser.parse_args().match_color,
        )
    elif parser.parse_args().model == "adain":
        final_img = run_adain(
            content_img_dir = parser.parse_args().content_img_dir, 
            style_img_dir = parser.parse_args().style_img_dir, 
            alpha= parser.parse_args().alpha_adain, 
            style_size=224, 
            apply_color_injection=parser.parse_args().apply_color_injection,
        )
        
    pred = predict(final_img, "models/model_rating.pth")
 #save the styled image
save_dir = f"output/{parser.parse_args().model}"
content_img_dir = parser.parse_args().content_img_dir
style_img_dir = parser.parse_args().style_img_dir
save_path = os.path.join(save_dir, f"{content_img_dir}_{style_img_dir}.jpg")

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
final_img.save(save_path)
