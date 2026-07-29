import torch.nn as nn
import torch.nn.functional as F
import math
import abc

import torch
import torch.nn as nn

from torch.autograd import Variable
from torchdiffeq import odeint, odeint_adjoint

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

def odesolver_init(func, z0, options=None):
    if options is None:
        Nt = 2
        method = 'Euler'  # Default method if options is None
    else:
        Nt = options['Nt']
        method = options.get('method', 'Euler')  # Default to 'Euler' if 'method' is not provided

    if method == 'Euler':
        solver = Euler(func, z0, Nt=Nt)
    elif method == 'RK2':
        solver = RK2(func, z0, Nt=Nt)
    elif method == 'RK4':
        solver = RK4(func, z0, Nt=Nt)
    else:
        print('Error: Unsupported method passed')
        return
    z1 = solver.integrate(z0)
    return z1


def flatten_params(params):
    flat_params = [p.contiguous().view(-1) for p in params]
    return torch.cat(flat_params) if len(flat_params) > 0 else torch.tensor([])


def flatten_params_grad(params, params_ref):
    _params = [p for p in params]
    _params_ref = [p for p in params_ref]
    flat_params = [p.contiguous().view(-1) if p is not None else torch.zeros_like(q).view(-1)
                   for p, q in zip(_params, _params_ref)]

    return torch.cat(flat_params) if len(flat_params) > 0 else torch.tensor([])


class Checkpointing_Adjoint(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args):
        z0, func, flat_params, options = args[0], args[1], args[2], args[3]
        ctx.func = func

        with torch.no_grad():
            ans = odesolver_init(func, z0, options)
        ctx.save_for_backward(z0)
        ctx.in1 = options
        return ans

    @staticmethod
    def backward(ctx, grad_output):
        z0 = ctx.saved_tensors
        options = ctx.in1
        func = ctx.func
        f_params = func.parameters()
        t = 0

        with torch.set_grad_enabled(True):
            z = Variable(z0[0].detach(), requires_grad=True)
            func_eval = odesolver_init(func, z, options)
            out1 = torch.autograd.grad(
                func_eval, z,
                grad_output, allow_unused=True, retain_graph=True)
            out2 = torch.autograd.grad(
                func_eval, f_params,
                grad_output, allow_unused=True, retain_graph=True)

        return out1[0], None, flatten_params_grad(out2, func.parameters()), None


def odesolver_adjoint(func, z0, options=None):
    flat_params = flatten_params(func.parameters())
    zs = Checkpointing_Adjoint.apply(z0, func, flat_params, options)

    return zs


def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)

class ODEBlock(nn.Module):

    def __init__(self, odefunc, Nt=2, method='euler'):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.options = {'method': method}
        self.method = method
        self.options.update({'Nt': int(Nt)})

    def forward(self, x, eval_times=None):
        if eval_times is None:
            integration_time = torch.tensor([0, 1]).float().type_as(x)
        else:
            integration_time = eval_times.type_as(x)

        # out = odesolver_adjoint(self.odefunc, x, self.options)
        out = odeint(self.odefunc, x, integration_time, method=self.method, rtol=1e-3, atol=1e-3)[-1]
        return out

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.nfe = 0  # 添加nfe属性，默认值为0
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BasicBlock2(nn.Module):
    expansion = 1

    def __init__(self, dim):
        super(BasicBlock2, self).__init__()
        in_planes = dim
        planes = dim
        stride = 1
        self.nfe = 0
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()

    def forward(self, t, x):
        self.nfe += 1
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, ODEBlock_=None, feature_dim=64):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.ODEBlock = ODEBlock_
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv_cifar10 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # 对于cifar10
        self.conv_BM = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 对于cifar10

        self.bn1 = nn.BatchNorm2d(64)
        self.layer1_1 = self._make_layer(64, num_blocks[0] - 1, stride=1)
        self.layer1_2 = self._make_layer2(64, num_blocks[0] - 1, stride=1)

        self.layer2_1 = self._make_layer(128, num_blocks[1] - 1, stride=2)
        self.layer2_2 = self._make_layer2(128, num_blocks[1] - 1, stride=1)

        self.layer3_1 = self._make_layer(256, num_blocks[2] - 1, stride=2)
        self.layer3_2 = self._make_layer2(256, num_blocks[2] - 1, stride=1)

        self.layer4_1 = self._make_layer(512, num_blocks[3] - 1, stride=2)
        self.layer4_2 = self._make_layer2(512, num_blocks[3] - 1, stride=1)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def _make_layer2(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(self.ODEBlock(BasicBlock2(self.in_planes)))
        return nn.Sequential(*layers)

    def forward(self, x):
        # out = F.relu(self.bn1(self.conv_BM(x)))
        # out = self.maxpool(out)  # 针对于224图像

        out = F.relu(self.bn1(self.conv_cifar10(x)))

        out = self.layer1_1(out)
        out = self.layer1_2(out)
        out = self.layer2_1(out)
        out = self.layer2_2(out)
        out = self.layer3_1(out)
        out = self.layer3_2(out)
        out = self.layer4_1(out)
        out = self.layer4_2(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    @property
    def nfe(self):
        nfe = 0
        for layer_name in ['layer1_2', 'layer2_2', 'layer3_2', 'layer4_2']:
            layer = getattr(self, layer_name)
            for block in layer:
                nfe += block.nfe
        return nfe / 4

    @nfe.setter
    def nfe(self, value):
        for layer_name in ['layer1_2', 'layer2_2', 'layer3_2', 'layer4_2']:
            layer = getattr(self, layer_name)
            for block in layer:
                block.nfe = value


def Get_AnodeV2_ResNet18(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], ODEBlock_=ODEBlock, num_classes=num_classes)


def Get_AnodeV2_ResNet34(num_classes):
    return ResNet(BasicBlock, [2, 3, 5, 2], ODEBlock_=ODEBlock, num_classes=num_classes)


def Get_AnodeV2_ResNet50(num_classes):
    return ResNet(BasicBlock, [3, 4, 6, 3], ODEBlock_=ODEBlock, num_classes=num_classes)


def lr_schedule(lr, epoch):
    optim_factor = 0
    if epoch > 250:
        optim_factor = 2
    elif epoch > 150:
        optim_factor = 1

    return lr / math.pow(10, (optim_factor))

