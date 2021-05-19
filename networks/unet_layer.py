import torch.nn.functional as F
import torch.nn as nn
import torch
import os

from .unet import UNet


PATH = '/mnt/d/TUM_courses/@DILab_Zeiss/FCAN-zeiss_edit/model/pretrained/unet.pth'

class UNet_Layer(nn.Module):
    def __init__(self,output_layer = None):
        super().__init__()
        self.pretrained = UNet(n_channels=1, n_classes=4)
        model_dict = self.pretrained.state_dict()
        #pretrained_dict = torch.load(PATH)
        pretrained_dict = torch.load(PATH, map_location=torch.device("cpu"))
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.pretrained.load_state_dict(model_dict)
        self.output_layer = output_layer
        self.layers = list(self.pretrained._modules.keys())
        self.layer_count = 0
        for l in self.layers:
            if l != self.output_layer:
                self.layer_count += 1
            else:
                break
        for i in range(1,len(self.layers)-self.layer_count):
            self.dummy_var = self.pretrained._modules.pop(self.layers[-i])
        
        self.net = nn.Sequential(self.pretrained._modules)
        self.pretrained = None

    def forward(self,x):
        x = self.net(x)
        return x