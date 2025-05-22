import numpy as np
from PIL import Image
import io
import random
import os
from generator import Generator
from model import StyleTransferModel
#initialize model
if __name__=="__main__":
    model_path="models/adain/model-ckp240.pth"
    model=StyleTransferModel(ckp=model_path).cuda()
    generator=Generator(model)
    data_dir="data/"
    content_dir=data_dir+"contents"
    style_dir=data_dir+"styles"
    output_dir="outputs/adain"
    generator.generate_batch(content_dir,style_dir,output_dir,s_size=None,c_size=None,c_size_ratio=0.5,s_size_ratio=0.5,alpha=1.0)