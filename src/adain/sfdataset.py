import torch
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
class SFDataset(Dataset):
    def __init__(self,pair_list,style_size,c_compose,s_compose):
        super().__init__()
        #self.content_imgs_list=[img_name.split(".")[0] for img_name in os.listdir(data_dir+"/contents")]
        #self.style_img_lists=[img_name.split(".")[0] for img_name in os.listdir(data_dir+"/styles")]
        self.pair_list=pair_list
        self.style_size=style_size
        self.c_compose=c_compose
        self.s_compose=s_compose
    def __getitem__(self,index):
        content_name,style_name=self.pair_list[index]
        content_image=np.array(Image.open(content_name).convert("RGB"))
        style_image=np.array(Image.open(style_name).convert("RGB"))
        # transform content and target images
        content_tensor=self.c_compose(image=content_image)["image"]
        #transform style images
        #style_image=A.Resize(self.style_size,self.style_size)(image=style_image)['image']
        style_tensor=self.s_compose(image=style_image)['image']
        content_name=content_name.split("/")[-1].split('.')[0]
        style_name=style_name.split("/")[-1].split('.')[0]
        return {
            "content": content_tensor,
            "style": style_tensor,
            "output_name":'_'.join([content_name,style_name])+'.jpg'
        }
    def __len__(self):
        return len(self.pair_list)