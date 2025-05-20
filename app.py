import streamlit as st
import numpy as np
from PIL import Image
import io
import torch
import random
import os
from src.adain.generator import Generator
from src.adain.model import StyleTransferModel
#initialize model
model_path="models/adain/model-ckp240.pth"
model=StyleTransferModel(ckp=model_path).cuda()
if model is None:
    st.error("Style transfer model not loaded. Please initialize the model.")
    st.stop()

generator = Generator(model, device="auto")

def style_transfer(content_image, style_image, style_size):
    """
    Generate a stylized image using the Generator class, matching content image size.
    
    Args:
        content_image (PIL.Image): Content image.
        style_image (PIL.Image): Style image.
        style_size (int): Style image preprocessing size.
    
    Returns:
        PIL.Image: Stylized image.
    """
    # Save temporary files for content and style images
    content_path = "temp_content.png"
    style_path = "temp_style.png"
    content_image.save(content_path)
    style_image.save(style_path)
    
    try:
        if style_size=="None":
            style_size=None
        # Generate stylized image
        stylized = generator.generate_single(
            content_path=content_path,
            style_path=style_path,
            s_size=style_size,
            alpha=1.0,
            retain_color=True
        )
        rating=compute_rating(stylized)
    finally:
        # Clean up temporary files
        if os.path.exists(content_path):
            os.remove(content_path)
        if os.path.exists(style_path):
            os.remove(style_path)
    
    return stylized,rating
def compute_rating(img):
    return random.randint(0,10)
def resize_image(image, max_size=300):
    """Resize image for display while maintaining aspect ratio."""
    image.thumbnail((max_size, max_size))
    return image

st.title("Style Transfer Demo")

# Layout with two columns for image uploads
col1, col2 = st.columns(2)

with col1:
    st.subheader("Content Image")
    content_file = st.file_uploader("Upload Content Image", type=["jpg", "jpeg", "png"], key="content")
    if content_file is not None:
        content_image = Image.open(content_file).convert("RGB")
        st.image(resize_image(content_image.copy()), caption="Content Image", use_container_width=True)
        st.write(f"Content Image Size: {content_image.size[0]}x{content_image.size[1]} pixels")
    else:
        content_image = None

with col2:
    st.subheader("Style Image")
    style_file = st.file_uploader("Upload Style Image", type=["jpg", "jpeg", "png"], key="style")
    if style_file is not None:
        style_image = Image.open(style_file).convert("RGB")
        st.image(resize_image(style_image.copy()), caption="Style Image", use_container_width=True)
    else:
        style_image = None

# Style size selection
st.subheader("Style Image Size")
style_size = st.selectbox("Select style image size (pixels)", ["None",50,150,300,500,700], index=0)

# Fusion button
if st.button("Fusion"):
    if content_image is None or style_image is None:
        st.error("Please upload both content and style images.")
    else:
        with st.spinner("Generating stylized image..."):
            try:
                # Generate stylized image
                stylized_image,rating = style_transfer(content_image, style_image, style_size)
                
                # Display the result
                st.subheader("Stylized Image")
                st.image(stylized_image, caption=f"Stylized Image ({content_image.size[0]}x{content_image.size[1]})", use_container_width=True)
                if rating is not None:
                    st.write(f"**Rating**: {rating}")
                else:
                    st.warning("Something wrong when trying to compute rating")
                # Option to download the result
                buffered = io.BytesIO()
                stylized_image.save(buffered, format="PNG")
                st.download_button(
                    label="Download Stylized Image",
                    data=buffered.getvalue(),
                    file_name="stylized_image.png",
                    mime="image/png"
                )
            except Exception as e:
                st.error(f"Error generating stylized image: {str(e)}")