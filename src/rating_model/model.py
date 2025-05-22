import torch.nn as nn
import torchvision.models as models

class RatingModel(nn.Module):
    def __init__(self):
        super(RatingModel, self).__init__()
        self.resnet = nn.Sequential(*list(models.resnet34(pretrained=True).children())[:-1])        
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.resnet.eval()
        
        self.flat = nn.Flatten()
        
        self.linear = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.resnet(x)
        x = self.flat(x)
        x = self.linear(x)
        
        return x