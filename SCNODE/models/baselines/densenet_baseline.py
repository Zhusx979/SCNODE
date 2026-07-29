import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义DenseNet中的DenseBlock
class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(BottleneckLayer(in_channels + i * growth_rate, growth_rate))

    def forward(self, x):
        for layer in self.layers:
            new_features = layer(x)
            x = torch.cat([x, new_features], dim=1)
        return x


# 定义DenseNet中的BottleneckLayer
class BottleneckLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(BottleneckLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return out


# 定义DenseNet中的过渡层（Transition Layer）
class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(F.relu(self.bn(x)))
        x = self.pool(x)
        return x


# 定义DenseNet模型
class DenseNetCustom(nn.Module):
    def __init__(self, num_classes=1000, num_blocks=[6, 12, 24, 16], growth_rate=32):
        super(DenseNetCustom, self).__init__()
        num_channels = 64  # 初始通道数
        self.conv1 = nn.Conv2d(3, num_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.blocks = nn.ModuleList()
        for i, num_layers in enumerate(num_blocks):
            block = DenseBlock(num_layers, num_channels, growth_rate)
            self.blocks.append(block)
            num_channels += num_layers * growth_rate
            if i != len(num_blocks) - 1:
                trans = TransitionLayer(num_channels, num_channels // 2)
                self.blocks.append(trans)
                num_channels //= 2

        self.bn_final = nn.BatchNorm2d(num_channels)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_channels, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        for block in self.blocks:
            x = block(x)
        x = self.avgpool(F.relu(self.bn_final(x)))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# 获取DenseNet模型
def Get_DenseNet(num_classes):
    return DenseNetCustom(num_classes=num_classes)

