import torch
import torch.nn as nn
import torch.nn.functional as F

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1,
            stride=1, padding=0, bias=bias
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, is_middle=False):
        super(ResidualBlock, self).__init__()
        self.is_middle = is_middle
        
        if is_middle:
            self.layers = nn.Sequential(
                nn.ReLU(inplace=False),
                SeparableConv2d(in_channels, out_channels, stride=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=False),
                SeparableConv2d(out_channels, out_channels, stride=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=False),
                SeparableConv2d(out_channels, out_channels, stride=1),
                nn.BatchNorm2d(out_channels),
            )
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.layers = nn.Sequential(
                nn.ReLU(inplace=False),
                SeparableConv2d(in_channels, out_channels, stride=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=False),
                SeparableConv2d(out_channels, out_channels, stride=1),
                nn.BatchNorm2d(out_channels),
                nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)
            )
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.layers(x)
        return out + residual

class Xception(nn.Module):
    def __init__(self, num_classes=2):
        super(Xception, self).__init__()
        # Entry flow
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.block1 = ResidualBlock(64, 128, stride=2)
        self.block2 = ResidualBlock(128, 256, stride=2)
        self.block3 = ResidualBlock(256, 728, stride=2)
        
        # Middle flow (8 blocks)
        self.middle_flow = nn.ModuleList([
            ResidualBlock(728, 728, is_middle=True) for _ in range(8)
        ])
        
        # Exit flow
        self.block4 = ResidualBlock(728, 1024, stride=2)
        
        self.sepconv1 = SeparableConv2d(1024, 1536)
        self.bn3 = nn.BatchNorm2d(1536)
        
        self.sepconv2 = SeparableConv2d(1536, 2048)
        self.bn4 = nn.BatchNorm2d(2048)
        
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        # Entry flow
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # Middle flow
        for block in self.middle_flow:
            x = block(x)
            
        # Exit flow
        x = self.block4(x)
        
        x = F.relu(self.bn3(self.sepconv1(x)))
        x = F.relu(self.bn4(self.sepconv2(x)))
        
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
