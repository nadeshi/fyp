import torch
import torch.nn as nn
from copy import deepcopy
from spikingjelly.activation_based import layer

class ResBlock(nn.Module):
    def __init__(self, in_channels, channels, downsample, spiking_neuron: callable = None, **kwargs):
        super().__init__()

        
        if downsample:
            self.conv1 = nn.Sequential(
                layer.Conv2d(in_channels, channels, kernel_size=3, stride=2, padding=1, bias=False),
                layer.BatchNorm2d(channels),
                spiking_neuron(**deepcopy(kwargs))
            )
           
            self.shortcut = nn.Sequential(
                layer.Conv2d(in_channels, channels, kernel_size=1, stride=2, bias=False),
                layer.BatchNorm2d(channels),
                spiking_neuron(**deepcopy(kwargs))
            )
        else:
            self.conv1 = nn.Sequential(
                layer.Conv2d(in_channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
                layer.BatchNorm2d(channels),
                spiking_neuron(**deepcopy(kwargs))
            )
            
            self.shortcut = nn.Sequential()        
        
        self.conv2 = nn.Sequential(
            layer.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            spiking_neuron(**deepcopy(kwargs))
        )
        
        self.sn1 = spiking_neuron(**deepcopy(kwargs))
        
    def forward(self, x: torch.Tensor):
        shortcut = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + shortcut
        x = self.sn1(x)
            
        return x
        
class DVSGestureNet(nn.Module):
    def __init__(self, channels=128, resblock = ResBlock, spiking_neuron: callable = None, **kwargs):
        super().__init__()
        
        
        self.layer0 = nn.Sequential(
            layer.Conv2d(2, channels, kernel_size=7, stride=2, padding=3, bias=False),
            layer.BatchNorm2d(channels),
            spiking_neuron(**deepcopy(kwargs)),
            layer.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        self.layer1 = nn.Sequential(
            resblock(128, 128, downsample=False, spiking_neuron=spiking_neuron, **kwargs),
            resblock(128, 128, downsample=False, spiking_neuron=spiking_neuron, **kwargs)
        )
          

        self.layer2 = nn.Sequential(
            resblock(128, 256, downsample=True, spiking_neuron=spiking_neuron, **kwargs),
            resblock(256, 256, downsample=False, spiking_neuron=spiking_neuron, **kwargs)
        )

        self.layer3 = nn.Sequential(
            resblock(256, 512, downsample=True, spiking_neuron=spiking_neuron, **kwargs),
            resblock(512, 512, downsample=False, spiking_neuron=spiking_neuron, **kwargs)
        )



        self.conv_fc = nn.Sequential(
            layer.AdaptiveAvgPool2d((1, 1)),
            layer.Flatten(),
            layer.Dropout(0.5),
            layer.Linear(channels * 2 * 2 , 110),
            spiking_neuron(**deepcopy(kwargs)),

            layer.VotingLayer(10)
        )

    def forward(self, x: torch.Tensor):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv_fc(x)

        return x
        
        
        
        
        
        
        
        
        