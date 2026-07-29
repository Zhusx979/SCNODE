import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint

# Step 1: 带 augment_dim 的 ConvODEFunc
class ConvODEFunc(nn.Module):
    def __init__(self, device, channels, augment_dim=0):
        super(ConvODEFunc, self).__init__()
        self.device = device
        self.channels = channels
        self.augment_dim = augment_dim

        # 增强后的通道数
        in_channels = channels + augment_dim

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, t, x):
        # x shape: (batch, channels, H, W)
        if self.augment_dim > 0:
            # 添加空白通道
            aug = torch.zeros(x.shape[0], self.augment_dim, x.shape[2], x.shape[3]).to(self.device)
            x = torch.cat([x, aug], dim=1)

        out = self.relu(self.bn1(self.conv1(x)))
        return out


# Step 2: 带 augment_dim 的 ODEBlock
class ODEBlock(nn.Module):
    def __init__(self, channels, augment_dim=0):
        super(ODEBlock, self).__init__()
        self.ode_func = ConvODEFunc(device='cuda', channels=channels, augment_dim=augment_dim)
        self.integration_time = torch.tensor([0., 1.]).float()

    def forward(self, x):
        self.integration_time = self.integration_time.to(x.device)
        out = odeint(self.ode_func, x, self.integration_time, rtol=1e-3, atol=1e-3)[-1]
        return out


# Step 3: ODEBasicBlock（ResNet Basic Block + ODE + augment_dim）
class ODEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, augment_dim=0):
        super(ODEBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.ode_block = ODEBlock(out_channels, augment_dim=augment_dim)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.ode_block(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# Step 4: ODEBottleneckBlock（ResNet Bottleneck Block + ODE + augment_dim）
class ODEBottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, augment_dim=0):
        super(ODEBottleneckBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.ode_block = ODEBlock(out_channels * self.expansion, augment_dim=augment_dim)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.ode_block(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# Step 5: 构建 ODEResNet 主体
class ODEResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, augment_dim=0):
        super(ODEResNet, self).__init__()
        self.in_channels = 64
        self.augment_dim = augment_dim
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], augment_dim=self.augment_dim)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, augment_dim=self.augment_dim)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, augment_dim=self.augment_dim)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, augment_dim=self.augment_dim)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, augment_dim=0):
        downsample = None
        if stride != 1 or self.in_channels != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, planes, stride, downsample, augment_dim=augment_dim))
        self.in_channels = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, planes, augment_dim=augment_dim))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


# Step 6: 快捷函数创建模型
def Get_aodenet18(num_classes=10, augment_dim=10):
    return ODEResNet(ODEBasicBlock, [2, 2, 2, 2], num_classes=num_classes, augment_dim=augment_dim)


def Get_aodenet34(num_classes=10, augment_dim=10):
    return ODEResNet(ODEBasicBlock, [3, 4, 6, 3], num_classes=num_classes, augment_dim=augment_dim)


def Get_aodenet50(num_classes=10, augment_dim=10):
    return ODEResNet(ODEBottleneckBlock, [3, 4, 6, 3], num_classes=num_classes, augment_dim=augment_dim)
