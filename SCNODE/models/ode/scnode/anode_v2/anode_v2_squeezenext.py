import abc
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def norm(dim):
    """
    创建组归一化层，通道数较小时使用较小的组数
    Args:
        dim (int): 输入通道数
    Returns:
        nn.GroupNorm: 组归一化层
    """
    return nn.GroupNorm(min(32, dim), dim)

class Time_Stepper(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, func, y0, Nt=2):
        self.func = func
        self.Nt = Nt

    @abc.abstractmethod
    def step(self, func, t, dt, y):
        pass

    def integrate(self, y0):
        y1 = y0
        dt = 1. / float(self.Nt)
        for n in range(self.Nt):
            t0 = 0 + n * dt
            y1 = self.step(self.func, t0, dt, y1)
        return y1

class Euler(Time_Stepper):
    def step(self, func, t, dt, y):
        out = y + dt * func(t, y)
        return out

class RK2(Time_Stepper):
    def step(self, func, t, dt, y):
        k1 = dt * func(t, y)
        k2 = dt * func(t + dt / 2.0, y + 1.0 / 2.0 * k1)
        out = y + k2
        return out

class RK4(Time_Stepper):
    def step(self, func, t, dt, y):
        k1 = dt * func(t, y)
        k2 = dt * func(t + dt / 2.0, y + 1.0 / 2.0 * k1)
        k3 = dt * func(t + dt / 2.0, y + 1.0 / 2.0 * k2)
        k4 = dt * func(t + dt, y + k3)
        out = y + 1.0 / 6.0 * k1 + 1.0 / 3.0 * k2 + 1.0 / 3.0 * k3 + 1.0 / 6.0 * k4
        return out

def odesolver(func, z0, options=None):
    if options is None:
        Nt = 2
        method = 'Euler'
    else:
        Nt = options.get('Nt', 2)
        method = options.get('method', 'Euler')

    if method == 'Euler':
        solver = Euler(func, z0, Nt=Nt)
    elif method == 'RK2':
        solver = RK2(func, z0, Nt=Nt)
    elif method == 'RK4':
        solver = RK4(func, z0, Nt=Nt)
    else:
        print('Error: Unsupported method passed')
        return None
    z1 = solver.integrate(z0)
    return z1

class ODEBlock(nn.Module):
    def __init__(self, odefunc, Nt=2):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.options = {}
        self.options.update({'Nt': int(Nt)})
        self.options.update({'method': 'Euler'})

    def forward(self, x):
        out = odesolver(self.odefunc, x, self.options)
        return out

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(BasicBlock, self).__init__()
        reduction = 0.5
        if 2 == stride:
            reduction = 1
        elif in_channels > out_channels:
            reduction = 0.25

        self.conv1 = nn.Conv2d(in_channels, int(in_channels * reduction), 1, stride, bias=True)
        self.bn1 = nn.BatchNorm2d(int(in_channels * reduction))
        self.conv2 = nn.Conv2d(int(in_channels * reduction), int(in_channels * reduction * 0.5), 1, 1, bias=True)
        self.bn2 = nn.BatchNorm2d(int(in_channels * reduction * 0.5))
        self.conv3 = nn.Conv2d(int(in_channels * reduction * 0.5), int(in_channels * reduction), (1, 3), 1, (0, 1), bias=True)
        self.bn3 = nn.BatchNorm2d(int(in_channels * reduction))
        self.conv4 = nn.Conv2d(int(in_channels * reduction), int(in_channels * reduction), (3, 1), 1, (1, 0), bias=True)
        self.bn4 = nn.BatchNorm2d(int(in_channels * reduction))
        self.conv5 = nn.Conv2d(int(in_channels * reduction), out_channels, 1, 1, bias=True)
        self.bn5 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if 2 == stride or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=True),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, input):
        output = F.relu(self.bn1(self.conv1(input)))
        output = F.relu(self.bn2(self.conv2(output)))
        output = F.relu(self.bn3(self.conv3(output)))
        output = F.relu(self.bn4(self.conv4(output)))
        output = F.relu(self.bn5(self.conv5(output)))
        output = output + F.relu(self.shortcut(input))  # 修复：移除原地操作
        output = F.relu(output)
        return output

class BasicBlock2(nn.Module):
    def __init__(self, dim):
        super(BasicBlock2, self).__init__()
        in_channels = dim
        out_channels = dim
        reduction = 0.5
        stride = 1
        self.nfe = 0

        self.conv1 = nn.Conv2d(in_channels, int(in_channels * reduction), 1, stride, bias=True)
        self.bn1 = nn.BatchNorm2d(int(in_channels * reduction))
        self.conv2 = nn.Conv2d(int(in_channels * reduction), int(in_channels * reduction * 0.5), 1, 1, bias=True)
        self.bn2 = nn.BatchNorm2d(int(in_channels * reduction * 0.5))
        self.conv3 = nn.Conv2d(int(in_channels * reduction * 0.5), int(in_channels * reduction), (1, 3), 1, (0, 1), bias=True)
        self.bn3 = nn.BatchNorm2d(int(in_channels * reduction))
        self.conv4 = nn.Conv2d(int(in_channels * reduction), int(in_channels * reduction), (3, 1), 1, (1, 0), bias=True)
        self.bn4 = nn.BatchNorm2d(int(in_channels * reduction))
        self.conv5 = nn.Conv2d(int(in_channels * reduction), out_channels, 1, 1, bias=True)
        self.bn5 = nn.BatchNorm2d(out_channels)

    def forward(self, t, x):
        self.nfe += 1
        output = F.relu(self.bn1(self.conv1(x)))
        output = F.relu(self.bn2(self.conv2(output)))
        output = F.relu(self.bn3(self.conv3(output)))
        output = F.relu(self.bn4(self.conv4(output)))
        output = F.relu(self.bn5(self.conv5(output)))
        return output

class SqueezeNext(nn.Module):
    def __init__(self, width_x, blocks, num_classes, ODEBlock_, feature_dim=64):
        super(SqueezeNext, self).__init__()
        self.in_channels = 64
        self.ODEBlock = ODEBlock_

        self.conv1 = nn.Conv2d(3, int(width_x * self.in_channels), 3, 1, 1, bias=True)  # For Cifar10
        self.bn1 = nn.BatchNorm2d(int(width_x * self.in_channels))

        self.initial_convs = nn.Sequential(
            nn.Conv2d(64, feature_dim, kernel_size=7, stride=2, padding=3, bias=True),
            # 224x224 -> 112x112
            norm(feature_dim),
            nn.ReLU(),  # 修复：移除 inplace=True
            nn.Conv2d(feature_dim, feature_dim, kernel_size=4, stride=2, padding=1, bias=True),
            # 112x112 -> 56x56
            norm(feature_dim),
            nn.ReLU(),  # 修复：移除 inplace=True
            nn.Conv2d(feature_dim, 64, kernel_size=4, stride=2, padding=1, bias=True),
            # 56x56 -> 28x28
            norm(64),
            nn.ReLU(),  # 修复：移除 inplace=True
            # nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=4, stride=2, padding=1, bias=True),  # 28x28 -> 14x14
            # norm(self.feature_dim),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=3, stride=2, padding=1, bias=True),  # 14x14 -> 7x7
            # norm(self.feature_dim),
            # nn.ReLU(inplace=True),
        )

        self.stage1_1 = self._make_layer1(1, width_x, 32, 1)
        self.stage1_2 = self._make_layer2(blocks[0] - 1, width_x, 32, 1)
        self.stage2_1 = self._make_layer1(1, width_x, 64, 2)
        self.stage2_2 = self._make_layer2(blocks[1] - 1, width_x, 64, 1)
        self.stage3_1 = self._make_layer1(1, width_x, 128, 2)
        self.stage3_2 = self._make_layer2(blocks[2] - 1, width_x, 128, 1)
        self.stage4_1 = self._make_layer1(1, width_x, 256, 2)
        self.stage4_2 = self._make_layer2(blocks[3] - 1, width_x, 256, 1)
        self.conv2 = nn.Conv2d(int(width_x * self.in_channels), int(width_x * 128), 1, 1, bias=True)
        self.bn2 = nn.BatchNorm2d(int(width_x * 128))
        self.linear = nn.Linear(int(width_x * 128), num_classes)

    def _make_layer1(self, num_block, width_x, out_channels, stride):
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        for _stride in strides:
            layers.append(BasicBlock(int(width_x * self.in_channels), int(width_x * out_channels), _stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def _make_layer2(self, num_block, width_x, out_channels, stride):
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        for _stride in strides:
            layers.append(self.ODEBlock(BasicBlock2(int(width_x * self.in_channels))))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, input):
        output = F.relu(self.bn1(self.conv1(input)))
        output = self.initial_convs(output)
        output = self.stage1_1(output)
        output = self.stage1_2(output)
        output = self.stage2_1(output)
        output = self.stage2_2(output)
        output = self.stage3_1(output)
        output = self.stage3_2(output)
        output = self.stage4_1(output)
        output = self.stage4_2(output)
        output = F.relu(self.bn2(self.conv2(output)))
        output = F.avg_pool2d(output, 4)
        output = output.view(output.size(0), -1)
        output = self.linear(output)
        return output

def SqNxt_23_1x(num_classes, ODEBlock):
    return SqueezeNext(1.0, [2, 2, 2, 2], num_classes, ODEBlock)

def lr_schedule(lr, epoch):
    optim_factor = 0
    if epoch > 250:
        optim_factor = 2
    elif epoch > 150:
        optim_factor = 1
    return lr / math.pow(10, (optim_factor))

def Get_AnodeV2_Sqnxt(num_classes):
    return SqNxt_23_1x(num_classes=num_classes, ODEBlock=ODEBlock)