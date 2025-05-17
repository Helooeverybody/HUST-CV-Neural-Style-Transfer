import numpy as np
from PIL import Image
import io
import random
import os
from generator import Generator
from model import StyleTransferModel
#initialize model
if __name__=="__main__":
    model_path="models/adain/model-ckp176.pth"
    model=StyleTransferModel(ckp=model_path).cuda()
    generator=Generator(model,color_retention_ratio=1)
    data_dir="data/"
    content_dir=data_dir+"contents"
    style_dir=data_dir+"styles"
    output_dir="outputs/adain"
    generator.generate(content_dir,style_dir,output_dir,alpha=0.3)