import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from src.adain.utils import compute_mean_std


class Encoder(nn.Module):
    """
    Feature Extractor based on VGG19
    """
    def __init__(self,pretrained=True,requires_grad=False):
        super().__init__()
        self.vgg=models.vgg19(pretrained=pretrained).features
        self.block1=self.vgg[:2]
        self.block2=self.vgg[2:7]
        self.block3=self.vgg[7:12]
        self.block4=self.vgg[12:21]
        self._set_grad(requires_grad)
    def _set_grad(self,requires_grad):
        for p in self.parameters():
            p.requires_grad=requires_grad
    def forward(self,x,return_last=True):
        x1=self.block1(x)
        x2=self.block2(x1)
        x3=self.block3(x2)
        x4=self.block4(x3)
        return x4 if return_last else (x1,x2,x3,x4)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        block1=nn.Sequential(
            nn.Conv2d(512,256,3,1,1,padding_mode="reflect"),
            nn.ReLU()
        )
        block2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, 1, 1, padding_mode="reflect"),
            nn.ReLU(),
        )

        block3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, 1, 1, padding_mode="reflect"),
            nn.ReLU(),
        )

        block4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, 1, 1, padding_mode="reflect"),
        )
        self.all=nn.ModuleList([block1,block2,block3,block4])
    def forward(self,x):
        for ix, module in enumerate(self.all):
            x=module(x)
            if ix < len(self.all)-1:
                x=F.interpolate(x,scale_factor=2,mode="nearest")
        return x

class AdaIn:
    """
    Adaptive Instance Normalization as proposed in
    'Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization' by Xun Huang, Serge Belongie.
    """
    def __call__(self,c_feature,s_feature,infer,eps=1e-8) -> torch.Tensor:

        c_mean,c_std=compute_mean_std(c_feature,infer=infer,eps=eps)
        s_mean,s_std=compute_mean_std(s_feature,infer=infer,eps=eps)
        output=(s_std * (c_feature-c_mean) / c_std) + s_mean
        return output

class StyleTransferModel(nn.Module):
    def __init__(self,ckp=None):
        super().__init__()
        self.encoder=Encoder()
        self.adain=AdaIn()
        self.decoder=Decoder()
        if ckp:
          self.decoder.load_state_dict(torch.load(ckp)['model'])
    def encoder_forward(self,x,return_last=False):
        return self.encoder(x,return_last=return_last)
    def generate(self,c_feats: torch.Tensor,s_feats: torch.Tensor,
                 alpha=1.0, infer=False):
        t=self.adain(c_feats,s_feats,infer)
        t=alpha*t + (1-alpha)*c_feats
        out=self.decoder(t)
        return (out,t)

    def forward(self,content_images:torch.Tensor,
                style_images:torch.Tensor,alpha=1.0,
                return_t=False,infer=False):
        c_feats=self.encoder(content_images,return_last=True)
        s_feats=self.encoder(style_images,return_last=True)
        out,t=self.generate(c_feats,s_feats,alpha,infer=infer)
        if infer:
            return out

        if return_t:
            return out,t
        else:
            return out