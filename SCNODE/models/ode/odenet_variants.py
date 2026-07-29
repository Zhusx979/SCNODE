import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint

class ODEBlock(nn.Module):
    def __init__(self, channels):
        super(ODEBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, t, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

class ODEBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ODEBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.ode_block = ODEBlock(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = odeint(self.ode_block, out, torch.tensor([0.0, 1.0]).float().to(out.device), rtol=1e-3, atol=1e-3,method='euler')[-1]
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ODEBottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ODEBottleneckBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
        self.relu = nn.ReLU(inplace=True)
        self.ode_block = ODEBlock(out_channels * 4)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn3(self.conv3(out))
        out = odeint(self.ode_block, out, torch.tensor([0.0, 1.0]).float().to(out.device), rtol=1e-3, atol=1e-3,method='euler')[-1]
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ODEResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ODEResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * (4 if block == ODEBottleneckBlock else 1), num_classes)
        self.register_buffer('t', torch.tensor([0.0, 1.0]))

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        expansion = 4 if block == ODEBottleneckBlock else 1
        if stride != 1 or self.in_channels != planes * expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * expansion),
            )
        layers = []
        layers.append(block(self.in_channels, planes, stride, downsample))
        self.in_channels = planes * expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def Get_odenet18(num_classes=1000):
    return ODEResNet(ODEBasicBlock, [2, 2, 2, 2], num_classes)

def Get_odenet34(num_classes=1000):
    return ODEResNet(ODEBasicBlock, [3, 4, 6, 3], num_classes)

def Get_odenet50(num_classes=1000):
    return ODEResNet(ODEBottleneckBlock, [3, 4, 6, 3], num_classes)
