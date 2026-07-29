import torch
import torch.nn as nn


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features  # 卷积层部分
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))  # 适应不同尺寸输入
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )  # 全连接层部分
        if init_weights:
            self._initialize_weights()  # 初始化权重

    def forward(self, x):
        x = self.features(x)  # 经过卷积层
        x = self.avgpool(x)  # 全局平均池化
        x = torch.flatten(x, 1)  # 拉平
        x = self.classifier(x)  # 经过全连接层
        return x

    def _initialize_weights(self):
        """使用标准初始化方法初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    """根据配置生成 VGG16 的卷积层部分"""
    layers = []
    in_channels = 3  # 输入通道数（RGB）
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers.extend([conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)])
            else:
                layers.extend([conv2d, nn.ReLU(inplace=True)])
            in_channels = v
    return nn.Sequential(*layers)


# VGG16 的网络结构配置
cfgs = {
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
}


def vgg16(num_classes=1000, batch_norm=False):
    """返回 VGG16 模型"""
    model = VGG(make_layers(cfgs["VGG16"], batch_norm=batch_norm), num_classes=num_classes)
    return model


