import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint  # 使用伴随方法降低显存占用


# 定义归一化函数（组归一化）
def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)


# 带时间维度的卷积层（拼接时间t）
class ConcatConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):

        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t  # 创建与x形状兼容的时间张量
        ttx = torch.cat([tt, x], 1)  # 在通道维度拼接时间和输入
        return self._layer(ttx)


# ODE函数定义
class ODEfunc(nn.Module):
    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        self.norm1 = norm(dim)  # 第一次归一化
        self.relu = nn.ReLU(inplace=True)  # ReLU激活函数
        self.conv1 = ConcatConv2d(dim, dim, ksize=3, stride=1, padding=1)  # 第一次卷积
        self.norm2 = norm(dim)  # 第二次归一化
        self.conv2 = ConcatConv2d(dim, dim, ksize=3, stride=1, padding=1)  # 第二次卷积
        self.norm3 = norm(dim)  # 第三次归一化
        self.nfe = 0  # 前向传播次数计数器

    def forward(self, t, x):
        self.nfe += 1  # 增加前向传播计数
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        out = self.norm3(out)
        return out


# ODE块定义
class ODEBlock(nn.Module):
    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()  # 积分时间范围 [0, 1]

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)  # 确保时间张量与输入类型一致
        out = odeint(self.odefunc, x, self.integration_time,method='dopri5', rtol=1e-3, atol=1e-3)  # ODE求解
        return out[1]  # 返回t=1时的输出

    @property
    def nfe(self):
        """获取前向传播次数"""
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        """设置前向传播次数"""
        self.odefunc.nfe = value


# 展平层
class Flatten(nn.Module):
    def __init__(self):
        """初始化展平层"""
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


# ODENet模型定义
class ODENet(nn.Module):
    def __init__(self, num_classes,feature_dim):
        super(ODENet, self).__init__()
        self.odeblock = ODEBlock(ODEfunc(feature_dim))
        self.norm = norm(feature_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = Flatten()
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(feature_dim, num_classes)
        self.downsample = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # Added padding=1 to maintain spatial dimensions
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            norm(64),  # Added normalization after last conv
            nn.ReLU(inplace=True)  # Added activation after last conv
        )

    def forward(self, x):
        out = self.downsample(x)
        out = self.odeblock(out)
        out = self.norm(out)
        out = self.relu(out)
        out = self.pool(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out


# 获取 ODENet 的函数
def Get_odenet(num_classes=1000):
    """
    创建一个适配 224x224 RGB 输入的 ODENet 模型
    Args:
        num_classes (int): 分类类别数，默认为 1000（适合 ImageNet）
    Returns:
        nn.Module: 配置好的 ODENet 模型
    """
    input_channels = 3  # RGB 图像
    feature_dim = 64  # 特征维度
    model = ODENet(num_classes, feature_dim = 64)
    return model