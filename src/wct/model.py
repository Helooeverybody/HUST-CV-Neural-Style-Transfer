import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import torchvision.transforms.v2 as transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm
import math
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os

conv5_1 = nn.Sequential(
    nn.Conv2d(3,3,(1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3,64,(3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64,64,(3, 3)),
    nn.ReLU(),
    nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64,128,(3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128,128,(3, 3)),
    nn.ReLU(),
    nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128,256,(3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256,256,(3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256,256,(3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256,256,(3, 3)),
    nn.ReLU(),
    nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256,512,(3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512,512,(3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512,512,(3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512,512,(3, 3)),
    nn.ReLU(),
    nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512,512,(3, 3)),
    nn.ReLU(),
)


class Encoder(nn.Module):
    def __init__(self, pretrained_path='models/conv5_1.pth'):
        super().__init__()
        self.net = conv5_1
        
        if pretrained_path is not None:
            self.net.load_state_dict(torch.load(pretrained_path, map_location=lambda storage, loc: storage))

    def forward(self, x, target):
        if target == 'relu1_1':
            return self.net[:4](x)
        elif target == 'relu2_1':
            return self.net[:11](x)
        elif target == 'relu3_1':
            return self.net[:18](x)
        elif target == 'relu4_1':
            return self.net[:31](x)
        elif target == 'relu5_1':
            return self.net(x)
        else:
            raise ValueError(f'target should be in ["relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1"] but not {target}')
        
        
        
import copy

dec5_1 = nn.Sequential( # Sequential,
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512,512,(3, 3)),
    nn.ReLU(),
    nn.UpsamplingNearest2d(scale_factor=2),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512,512,(3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512,512,(3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512,512,(3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512,256,(3, 3)),
    nn.ReLU(),
    nn.UpsamplingNearest2d(scale_factor=2),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256,256,(3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256,256,(3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256,256,(3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256,128,(3, 3)),
    nn.ReLU(),
    nn.UpsamplingNearest2d(scale_factor=2),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128,128,(3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128,64,(3, 3)),
    nn.ReLU(),
    nn.UpsamplingNearest2d(scale_factor=2),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64,64,(3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64,3,(3, 3)),
)


class Decoder(nn.Module):
    def __init__(self, level, pretrained_path=None):
        super().__init__()
        if level == 1:
            self.net = nn.Sequential(*copy.deepcopy(list(dec5_1.children())[-2:]))
        elif level == 2:
            self.net = nn.Sequential(*copy.deepcopy(list(dec5_1.children())[-9:]))
        elif level == 3:
            self.net = nn.Sequential(*copy.deepcopy(list(dec5_1.children())[-16:]))
        elif level == 4:
            self.net = nn.Sequential(*copy.deepcopy(list(dec5_1.children())[-29:]))
        elif level == 5:
            self.net = dec5_1
            
        if pretrained_path is not None:
            self.net.load_state_dict(torch.load(pretrained_path, map_location=lambda storage, loc: storage))

    def forward(self, x):
        return self.net(x)
    
    
    
def sqrt_matrix(mtx):
    size = mtx.size()
    u, e, v = torch.svd(mtx, some=False)
    k_c = size[0]
    for i in range(size[0]):
        if e[i] < 0.00001:
            k_c = i
            break
    d = e[:k_c].pow(0.5)
    m_step1 = torch.mm(v[:, :k_c], torch.diag(d))
    m = torch.mm(m_step1, v[:, :k_c].t())
    return m

def sqrt_inv_matrix(mtx):
    size = mtx.size()
    u, e, v = torch.svd(mtx, some=False)
    k_c = size[0]
    for i in range(size[0]):
        if e[i] < 0.00001:
            k_c = i
            break
    d = e[:k_c].pow(-0.5)
    m_step1 = torch.mm(v[:, :k_c], torch.diag(d))
    m = torch.mm(m_step1, v[:, :k_c].t())
    return m
    

def feature_transform(content_feature, style_feature, alpha=1.0):
    content_feature = content_feature.type(dtype=torch.float64)
    style_feature = style_feature.type(dtype=torch.float64)
    
    content_feature1 = content_feature.squeeze(0)
    cDim = content_feature1.size()
    content_feature1 = content_feature1.reshape(cDim[0], -1)
    c_mean = torch.mean(content_feature1, 1, keepdim=True)
    content_feature1 = content_feature1 - c_mean
    content_cov = torch.mm(content_feature1, content_feature1.t()).div(cDim[1]*cDim[2]-1)
    
    style_feature1 = style_feature.squeeze(0)
    sDim = style_feature1.size()
    style_feature1 = style_feature1.reshape(sDim[0], -1)
    s_mean = torch.mean(style_feature1, 1, keepdim=True)
    style_feature1 = style_feature1 - s_mean
    style_cov = torch.mm(style_feature1, style_feature1.t()).div(sDim[1]*sDim[2]-1)
    
    sqrtInvU = sqrt_inv_matrix(content_cov)
    sqrtU = sqrt_matrix(content_cov)
    C = torch.mm(torch.mm(sqrtU, style_cov), sqrtU)
    sqrtC = sqrt_matrix(C)
    T = torch.mm(torch.mm(sqrtInvU, sqrtC), sqrtInvU)
    target_feature = torch.mm(T, content_feature1)
    target_feature = target_feature + s_mean
    res_feature = target_feature.reshape(cDim[0], cDim[1], cDim[2]).unsqueeze(0).float()
    
    res_feature = alpha * res_feature + (1.0 - alpha) * content_feature
    return res_feature.type(dtype=torch.float32)


class MultiLevelAE_OST(nn.Module):
    def __init__(self, pretrained_path_dir='models'):
        super().__init__()
        self.encoder = Encoder(f'{pretrained_path_dir}/conv5_1.pth')
        self.decoder1 = Decoder(1, f'{pretrained_path_dir}/dec1_1.pth')
        self.decoder2 = Decoder(2, f'{pretrained_path_dir}/dec2_1.pth')
        self.decoder3 = Decoder(3, f'{pretrained_path_dir}/dec3_1.pth')
        self.decoder4 = Decoder(4, f'{pretrained_path_dir}/dec4_1.pth')
        self.decoder5 = Decoder(5, f'{pretrained_path_dir}/dec5_1.pth')

    def transform_level(self, content_image, style_image, alpha, level):
        content_feature = self.encoder(content_image, f'relu{level}_1')
        style_feature = self.encoder(style_image, f'relu{level}_1')
        res = feature_transform(content_feature, style_feature, alpha)
        return getattr(self, f'decoder{level}')(res)

    def forward(self, content_image, style_image, alpha=1):
        r5 = self.transform_level(content_image, style_image, alpha, 5)
        r4 = self.transform_level(r5, style_image, alpha, 4)
        r3 = self.transform_level(r4, style_image, alpha, 3)
        r2 = self.transform_level(r3, style_image, alpha, 2)
        r1 = self.transform_level(r2, style_image, alpha, 1)

        return r1