import streamlit as st
import numpy as np
from PIL import Image
import io
import torch
import random
import os
from infer import *
import gc
# Set page configuration for wide mode
st.set_page_config(layout="wide")

def style_transfer(c_name, s_name, style_size_ratio, model_type, alpha=1.0, retain_color=True):
    if model_type == "adain":
        stylized = run_adain(c_name, s_name, retain_color=retain_color,
                             alpha=alpha, c_size_ratio=0.5,
                             s_size_ratio=style_size_ratio,
                             )
    elif model_type == "wct":
        stylized = run_wct(c_name, s_name, retain_color=retain_color,
                           content_size_mult=0.5,
                           style_size_mult=style_size_ratio, alpha=alpha,
                           )
    elif model_type == "patch_st":
        stylized = run_patch_st(c_name, s_name,retain_color=retain_color,
                                resize_percent_content=50,
                                resize_percent_style=style_size_ratio*100,
                                )
    elif model_type == "transformer":
        stylized = run_transformer(c_name, s_name, 
                                   retain_color=retain_color,
                                   )
    else:
        raise ValueError("Invalid model type")
    rating = predict(stylized, "models/model_rating.pth")
    return stylized, round(rating,3)

def resize_to_same_height(image1, image2, target_height=350):
    """
    Resize two images to the same height while preserving aspect ratio.
    """
    # Calculate aspect ratios
    aspect_ratio1 = image1.size[0] / image1.size[1]
    aspect_ratio2 = image2.size[0] / image2.size[1] if len(image2.size) > 1 else 1

    # Resize both images to the target height
    new_height = target_height
    new_width1 = int(new_height * aspect_ratio1)
    new_width2 = int(new_height * aspect_ratio2)

    resized_image1 = image1.resize((new_width1, new_height), Image.Resampling.LANCZOS)
    resized_image2 = image2.resize((new_width2, new_height), Image.Resampling.LANCZOS)

    return resized_image1, resized_image2

# Sidebar for selections and adjustments
with st.sidebar:
    st.subheader("Content Image")
    content_file = st.file_uploader("Upload Content Image", type=["jpg", "jpeg", "png"], key="content")
    if content_file is not None:
        c_name = content_file.name.split(".")[0]
        content_image = Image.open(content_file).convert("RGB")
    else:
        content_image = None

    st.subheader("Style Image")
    style_file = st.file_uploader("Upload Style Image", type=["jpg", "jpeg", "png"], key="style")
    if style_file is not None:
        s_name = style_file.name.split(".")[0]
        style_image = Image.open(style_file).convert("RGB")
    else:
        style_image = None

    st.subheader("Model Selection")
    model_type = st.selectbox("Select Model", ["adain", "wct", "patch_st","transformer", "all"], index=0)

    st.subheader("Style Image Size")
    style_size_ratio = st.slider("Select style image size ", 0.1, 1.0, 0.5, 0.1)

    # Alpha slider for wct and adain
    alpha = 1.0
    if model_type in ["wct", "adain","all"]:
        alpha = st.slider("Alpha (Style-Content Balance)", 0.0, 1.0, 1.0, 0.1)

    st.subheader("Preserve content image color")
    retain_color = st.checkbox("Preserve color?", value=True)

# Main area for image display and Fusion button
st.title("Style Transfer Demo")

# Initial preview in three columns (third column empty)
preview_col1, preview_col2, preview_col3 = st.columns(3)

if 'content_image' in locals() and content_image is not None:
    with preview_col1:
        st.subheader("Content Image")
        resized_content = content_image.copy()
        if 'style_image' in locals() and style_image is not None:
            resized_content, resized_style = resize_to_same_height(content_image.copy(), style_image.copy())
        else:
            resized_content, _ = resize_to_same_height(content_image.copy(), Image.new("RGB", (1, 1)))
        st.image(resized_content, caption=f"Content ({content_image.size[0]}x{content_image.size[1]})")

if 'style_image' in locals() and style_image is not None:
    with preview_col2:
        st.subheader("Style Image")
        # Already resized in the previous block if content_image exists
        st.image(resized_style, caption=f"Style ({style_image.size[0]}x{style_image.size[1]})")

# Fusion button in the main area
if 'content_image' in locals() and 'style_image' in locals() and content_image is not None and style_image is not None:
    if st.button("Fusion"):
        with st.spinner("Generating stylized image..."):
            try:
                if model_type == "all":

                    # New three columns for all results
                    results_col1, results_col2, results_col3, results_col4 = st.columns(4)
                    results = {}
                    for mt in ["adain", "wct", "patch_st", "transformer"]:
                        stylized_image, rating = style_transfer(c_name, s_name, 
                                                                style_size_ratio, mt, 
                                                                alpha=alpha, retain_color=retain_color)
                        results[mt] = (stylized_image, rating)
                        # Clear memory after each model run
                        del stylized_image
                        torch.cuda.empty_cache()
                        gc.collect()
                    resized_adain, resized_wct = resize_to_same_height(results["adain"][0], results["wct"][0])
                    resized_patchst, resized_transformer = resize_to_same_height(results["patch_st"][0], results["transformer"][0])
                   

                    with results_col1:
                        st.subheader("AdaIN Output")
                        st.image(resized_adain, caption=f"AdaIN ({content_image.size[0]}x{content_image.size[1]}")
                        st.write(f"Rating: {results['adain'][1]}")
                        buffered = io.BytesIO()
                        results["adain"][0].save(buffered, format="PNG")
                        st.download_button(label="Download AdaIN", data=buffered.getvalue(), file_name=f"adain_{c_name}___{s_name}.png", mime="image/png")
                    
                    with results_col2:
                        st.subheader("WCT Output")
                        st.image(resized_wct, caption=f"WCT ({content_image.size[0]}x{content_image.size[1]}")
                        st.write(f"Rating: {results['wct'][1]}")
                        buffered = io.BytesIO()
                        results["wct"][0].save(buffered, format="PNG")
                        st.download_button(label="Download WCT", data=buffered.getvalue(), file_name=f"wct_{c_name}___{s_name}.png", mime="image/png")
                    
                    with results_col3:
                        st.subheader("PatchST Output")
                        st.image(resized_patchst, caption=f"PatchST ({content_image.size[0]}x{content_image.size[1]}")
                        st.write(f" Rating: {results['patch_st'][1]}")
                        buffered = io.BytesIO()
                        results["patch_st"][0].save(buffered, format="PNG")
                        st.download_button(label="Download PatchST", data=buffered.getvalue(), file_name=f"patch_st_{c_name}___{s_name}.png", mime="image/png")
                    with results_col4:
                        st.subheader("Transformer Output")
                        st.image(resized_transformer, caption=f"Transformer ({content_image.size[0]}x{content_image.size[1]}")
                        st.write(f"Rating: {results['transformer'][1]}")
                        buffered = io.BytesIO()
                        results["transformer"][0].save(buffered, format="PNG")
                        st.download_button(label="Download Transformer", data=buffered.getvalue(), file_name=f"transformer_{c_name}___{s_name}.png", mime="image/png")
                else:
                    # Use the same three columns for preview and result
                    with preview_col3:
                        stylized_image, rating = style_transfer(c_name, s_name, 
                                                                style_size_ratio, model_type, 
                                                                alpha=alpha, retain_color=retain_color)
                        st.subheader("Stylized Image")
                        resized_stylized, _ = resize_to_same_height(stylized_image, stylized_image)
                        st.image(resized_stylized, caption=f"{model_type.upper()} ({content_image.size[0]}x{content_image.size[1]}")
                        if rating is not None:
                            st.write(f"**Rating**: {rating}")
                        else:
                            st.warning("Something wrong when trying to compute rating")
                        buffered = io.BytesIO()
                        stylized_image.save(buffered, format="PNG")
                        st.download_button(
                            label="Download Stylized Image",
                            data=buffered.getvalue(),
                            file_name=f"{model_type}_{c_name}___{s_name}.png",
                            mime="image/png"
                        )
                         # Clear memory after generation
                        del stylized_image
                        torch.cuda.empty_cache()
                        gc.collect()
            except Exception as e:
                st.error(f"Error generating stylized image: {str(e)}")
else:
    st.info("Please upload both content and style images to proceed.")