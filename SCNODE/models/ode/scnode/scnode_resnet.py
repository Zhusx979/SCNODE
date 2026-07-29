import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import abc

from torchdiffeq import odeint, odeint_adjoint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.autograd import Variable

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

def odesolver_init(func, z0, options = None):
    if options == None:
        Nt = 2
    else:
        Nt = options['Nt']
    if (options['method'] == 'Euler'):
        solver = Euler(func, z0, Nt = Nt)
    elif (options['method'] == 'RK2'):
        solver = RK2(func, z0, Nt = Nt)
    elif (options['method'] == 'RK4'):
        solver = RK4(func, z0, Nt = Nt)
    else:
        print('error unsupported method passed')
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
        z0, func, flat_params, options= args[0], args[1], args[2], args[3]
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
            z = Variable(z0[0].detach(),requires_grad=True)
            func_eval = odesolver_init(func, z, options)
            out1 = torch.autograd.grad(
               func_eval,  z,
               grad_output, allow_unused=True, retain_graph=True)
            out2 = torch.autograd.grad(
               func_eval,  f_params,
               grad_output, allow_unused=True, retain_graph=True)

        return out1[0], None, flatten_params_grad(out2, func.parameters()), None



def odesolver_adjoint(func, z0, options = None):

    flat_params = flatten_params(func.parameters())
    zs = Checkpointing_Adjoint.apply(z0, func, flat_params, options)

    return zs


def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)



# grok
class TABN(nn.Module):
    """
    Temporal Adaptive Batch Normalization (TA-BN) for Neural ODEs.

    This module replaces standard BatchNorm in Neural ODEs. It maintains time-dependent
    population statistics and learnable parameters on a fixed grid T* = linspace(0, T, num_grids).
    For 4D inputs (B, C, H, W), normalization is channel-wise (stats over B, H, W).

    Args:
        num_features (int): Number of channels C.
        num_grids (int): Number of time grids M+1 (default 101 for T=1.0).
        eps (float): Epsilon for numerical stability.
        momentum (float): Momentum η for updating running stats (default 0.1).
        T (float): Total time span (default 1.0, common in Neural ODEs).
        affine (bool): If True, learn gamma and beta (default True).

    Note: running_mean/var: (num_grids, C)
          gamma/beta: (num_grids, C) if affine.
    """

    def __init__(self, num_features, num_grids=101, eps=1e-5, momentum=0.1, T=1.0, affine=True):
        super(TABN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.T = T
        self.affine = affine
        self.num_grids = num_grids
        self.register_buffer('T_star',
                             torch.linspace(0, T, num_grids, device='cuda' if torch.cuda.is_available() else 'cpu'))

        # Running stats: shape (num_grids, C)
        self.register_buffer('running_mean', torch.zeros(num_grids, num_features))
        self.register_buffer('running_var', torch.ones(num_grids, num_features))

        if affine:
            self.gamma = nn.Parameter(torch.ones(num_grids, num_features))
            self.beta = nn.Parameter(torch.zeros(num_grids, num_features))
        else:
            self.register_buffer('gamma', torch.ones(num_grids, num_features))
            self.register_buffer('beta', torch.zeros(num_grids, num_features))

    def forward(self, x, t, training=False):
        """
        Forward pass at time t (scalar).

        Args:
            x (torch.Tensor): Input (B, C, H, W)
            t (float): Current time tj
            training (bool): Training mode flag

        Returns:
            torch.Tensor: Normalized x
        """
        B, C, H, W = x.shape
        device = x.device
        t = t.to(device)

        # Step 1: Find l such that T_star[l] <= t < T_star[l+1]
        l = torch.searchsorted(self.T_star, t).item() - 1  # Largest l where T_star[l] < t
        if l < 0:
            l = 0
        if l >= self.num_grids - 1:
            l = self.num_grids - 2
        t_l = self.T_star[l]
        t_lp1 = self.T_star[l + 1]

        # Step 2: Compute interpolation weights
        denom = t_lp1 - t_l
        omega1 = (t_lp1 - t) / denom
        omega2 = (t - t_l) / denom

        # Channel-wise means/vars over (B, H, W): shape (C,)
        if training:
            # Step 4: Mini-batch stats
            mu_j = torch.mean(x, dim=[0, 2, 3], keepdim=False)  # (C,)
            var_j = torch.var(x, dim=[0, 2, 3], keepdim=False, unbiased=False)  # (C,)

            # Step 5: Interpolate gamma/alpha (beta)
            gamma_j = omega1 * self.gamma[l] + omega2 * self.gamma[l + 1]  # (C,)
            beta_j = omega1 * self.beta[l] + omega2 * self.beta[l + 1]  # (C,)

            # Steps 6-7: Update running stats (weighted moving average)
            self.running_mean[l] = (1 - self.momentum * omega1) * self.running_mean[l] + self.momentum * omega1 * mu_j
            self.running_mean[l + 1] = (1 - self.momentum * omega2) * self.running_mean[
                l + 1] + self.momentum * omega2 * mu_j
            self.running_var[l] = (1 - self.momentum * omega1) * self.running_var[l] + self.momentum * omega1 * var_j
            self.running_var[l + 1] = (1 - self.momentum * omega2) * self.running_var[
                l + 1] + self.momentum * omega2 * var_j

            mu_j_full = mu_j.view(1, C, 1, 1)  # For broadcasting
            var_j_full = var_j.view(1, C, 1, 1)
            sigma_j = torch.sqrt(var_j_full + self.eps)
        else:
            # Step 9: Interpolate population stats
            mu_j = omega1 * self.running_mean[l] + omega2 * self.running_mean[l + 1]  # (C,)
            var_j = omega1 * self.running_var[l] + omega2 * self.running_var[l + 1]  # (C,)

            # Step 10: Interpolate gamma/alpha
            gamma_j = omega1 * self.gamma[l] + omega2 * self.gamma[l + 1]  # (C,)
            beta_j = omega1 * self.beta[l] + omega2 * self.beta[l + 1]  # (C,)

            mu_j_full = mu_j.view(1, C, 1, 1)  # For broadcasting
            var_j_full = var_j.view(1, C, 1, 1)
            sigma_j = torch.sqrt(var_j_full + self.eps)

        # Step 12: Normalize and scale/shift
        x_centered = x - mu_j_full
        x_norm = x_centered / sigma_j
        gamma_j_full = gamma_j.view(1, C, 1, 1)
        beta_j_full = beta_j.view(1, C, 1, 1)
        out = x_norm * gamma_j_full + beta_j_full

        return out


import torch
import torch.nn as nn
import torch.nn.functional as F


class TWBN(nn.Module):
    """
    Temporal Window BatchNorm (TW-BN) for Neural ODEs.

    This module reduces the computational overhead by introducing a sliding time window approach
    and performing optimized batch normalization with temporal statistics.

    Args:
        num_features (int): Number of channels C.
        window_size (int): Number of time grids in the sliding window (default 5).
        eps (float): Epsilon for numerical stability.
        momentum (float): Momentum η for updating running stats (default 0.1).
        T (float): Total time span (default 1.0).
        affine (bool): If True, learn gamma and beta (default True).
    """

    def __init__(self, num_features, window_size=5, eps=1e-5, momentum=0.1, T=1.0, affine=True):
        super(TWBN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.T = T
        self.affine = affine
        self.window_size = window_size

        # T_star initialization with a fixed grid (time points)
        self.register_buffer('T_star',
                             torch.linspace(0, T, window_size, device='cuda' if torch.cuda.is_available() else 'cpu'))

        # Running stats initialization: (window_size, C)
        self.register_buffer('running_mean', torch.zeros(window_size, num_features))
        self.register_buffer('running_var', torch.ones(window_size, num_features))

        if affine:
            self.gamma = nn.Parameter(torch.ones(window_size, num_features))
            self.beta = nn.Parameter(torch.zeros(window_size, num_features))
        else:
            self.register_buffer('gamma', torch.ones(window_size, num_features))
            self.register_buffer('beta', torch.zeros(window_size, num_features))

    def _find_window(self, t, T_star):
        """Find the window of time grids based on the time t."""
        l = torch.searchsorted(T_star, t).item() - 1  # Find the largest l where T_star[l] < t
        l = max(0, min(l, len(T_star) - self.window_size))  # Keep within valid window range
        return l

    def _smooth_stats(self, stats, l):
        """Apply smoothing to the stats over the sliding window."""
        # We use a simple moving average with weights based on proximity to t
        window_size = self.window_size
        weights = torch.linspace(1, 0.1, window_size, device=stats.device)  # Decreasing weights for older grids
        smoothed_stats = torch.sum(stats[l:l + window_size] * weights.view(-1, 1), dim=0) / torch.sum(weights)
        return smoothed_stats

    def _interpolate(self, values, omega1, omega2, l):
        """Perform linear interpolation between l and l+1."""
        return omega1 * values[l] + omega2 * values[l + 1]

    def forward(self, x, t, training=False):
        """
        Forward pass with time window-based batch normalization.

        Args:
            x (torch.Tensor): Input (B, C, H, W)
            t (float): Current time tj
            training (bool): Mode flag

        Returns:
            torch.Tensor: Normalized x
        """
        B, C, H, W = x.shape
        device = x.device
        t = t.to(device)

        # Step 1: Find the sliding window index for time t
        l = self._find_window(t.item(), self.T_star)
        t_l, t_lp1 = self.T_star[l], self.T_star[l + 1]

        # Step 2: Compute interpolation weights
        denom = t_lp1 - t_l
        omega1 = (t_lp1 - t) / denom
        omega2 = (t - t_l) / denom

        if training:
            # Step 3: Calculate mini-batch stats (mean and variance)
            mu_j = torch.mean(x, dim=[0, 2, 3], keepdim=False)  # Mean (C,)
            var_j = torch.var(x, dim=[0, 2, 3], keepdim=False, unbiased=False)  # Variance (C,)

            # Step 4: Smooth the stats using the time window
            mu_j_smooth = self._smooth_stats(mu_j, l)
            var_j_smooth = self._smooth_stats(var_j, l)

            # Step 5: Interpolate gamma and beta using the current time weights
            gamma_j = self._interpolate(self.gamma, omega1, omega2, l)
            beta_j = self._interpolate(self.beta, omega1, omega2, l)

            # Step 6: Update running statistics with time-dependent momentum
            self.running_mean[l] = (1 - self.momentum * omega1) * self.running_mean[
                l] + self.momentum * omega1 * mu_j_smooth
            self.running_var[l] = (1 - self.momentum * omega1) * self.running_var[
                l] + self.momentum * omega1 * var_j_smooth

            mu_j_full = mu_j_smooth.view(1, C, 1, 1)
            var_j_full = var_j_smooth.view(1, C, 1, 1)
            sigma_j = torch.sqrt(var_j_full + self.eps)
        else:
            # Step 7: Interpolate population stats during inference
            mu_j = self._interpolate(self.running_mean, omega1, omega2, l)
            var_j = self._interpolate(self.running_var, omega1, omega2, l)
            gamma_j = self._interpolate(self.gamma, omega1, omega2, l)
            beta_j = self._interpolate(self.beta, omega1, omega2, l)

            mu_j_full = mu_j.view(1, C, 1, 1)
            var_j_full = var_j.view(1, C, 1, 1)
            sigma_j = torch.sqrt(var_j_full + self.eps)

        # Step 8: Normalize the input tensor and apply scale/shift
        x_centered = x - mu_j_full
        x_norm = x_centered / sigma_j
        gamma_j_full = gamma_j.view(1, C, 1, 1)
        beta_j_full = beta_j.view(1, C, 1, 1)
        out = x_norm * gamma_j_full + beta_j_full

        return out


class Conv2dTime(nn.Conv2d):
    def __init__(self, in_channels, *args, **kwargs):
        super(Conv2dTime, self).__init__(in_channels + 1, *args, **kwargs)

    def forward(self, t, x):
        t_img = torch.ones_like(x[:, :1, :, :]) * t
        t_and_x = torch.cat([t_img, x], 1)
        return super(Conv2dTime, self).forward(t_and_x)

class ODEBlock(nn.Module):
    def __init__(self, odefunc, Nt=2, method='euler', augment_dim=1):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.augment_dim = augment_dim
        self.options = {'Nt': int(Nt), 'method': method}  # 初始化 method 属性
        self.method = method  # 将 method 属性单独赋值

    def forward(self, x, eval_times=None):
        if eval_times is None:
            integration_time = torch.tensor([0, 1]).float().type_as(x)
        else:
            integration_time = eval_times.type_as(x)


        # 添加 augment_dim 个零通道
        if self.augment_dim > 0:
            batch_size, channels, height, width = x.shape
            aug = torch.zeros(batch_size, self.augment_dim, height, width, device=x.device, dtype=x.dtype)
            x = torch.cat([x, aug], dim=1)  # 通道数变为 dim + augment_dim

        # out = odeint_adjoint(self.odefunc, x, integration_time, method=self.method,rtol=1e-3, atol=1e-3)[-1]
        out = odeint(self.odefunc, x, integration_time, method=self.method,rtol=1e-3, atol=1e-3)[-1]
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
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BasicBlock2(nn.Module):
    expansion = 1

    def __init__(self, dim, num_filters=64,augment_dim=1):
        super(BasicBlock2, self).__init__()
        self.nfe = 0
        self.dim = dim
        self.in_planes = dim
        self.augment_dim = augment_dim
        self.total_in_planes = dim+self.augment_dim
        self.conv1 = Conv2dTime(self.total_in_planes, num_filters,
                                kernel_size=1, stride=1, padding=0)

        # self.bn1 = nn.BatchNorm2d(num_filters)
        self.bn1 = TWBN(num_filters)
        self.conv2 = Conv2dTime(num_filters, num_filters,
                                kernel_size=3, stride=1, padding=1)

        # self.bn2 = nn.BatchNorm2d(num_filters)
        self.bn2 = TWBN(num_filters)
        self.shortcut = nn.Sequential()

    def forward(self, t, x):
        self.nfe += 1
        # 这里的x已经加上了augment_dim个零通道，为了残差连接，num_filters 必须= dim + augment_dim
        out = self.conv1(t,x)
        out = F.relu(self.bn1(out,t))
        out = self.conv2(t,out)
        out = self.bn2(out,t)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, ODEBlock_=ODEBlock,  augment_dim=1):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.ODEBlock = ODEBlock_
        self.augment_dim = augment_dim
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv_cifar10 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)#对于cifar10
        self.conv_BM = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)#对于cifar10
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Layer 1
        self.layer1 = self._make_layer(out_planes=64, num_blocks=num_blocks[0] - 1, stride=1)
        # Layer 2
        self.layer2 = self._make_layer(out_planes=128, num_blocks=num_blocks[1] - 1, stride=2)
        # Layer 3
        self.layer3 = self._make_layer(out_planes=256, num_blocks=num_blocks[2] - 1, stride=2)
        # Layer 4
        self.layer4 = self._make_layer(out_planes=512, num_blocks=num_blocks[3] - 1, stride=2)
        # Linear layer
        self.linear = nn.Linear((512+self.augment_dim) * block.expansion, num_classes)
    def _make_layer(self, out_planes, num_blocks, stride):
        layers = []
        strides = [stride] + [1] * (num_blocks - 1)
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, out_planes, stride))
            self.in_planes = out_planes * BasicBlock.expansion
            current_out_planes = out_planes + self.augment_dim #决定了num_filters
            block = BasicBlock2(self.in_planes, num_filters=current_out_planes,augment_dim=self.augment_dim)
            layers.append(self.ODEBlock(block,augment_dim=self.augment_dim))
            self.in_planes = current_out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv_BM(x)))
        out = self.maxpool(out)#针对于224图像

        # out = F.relu(self.bn1(self.conv_cifar10(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = F.avg_pool2d(out, 4)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    @property
    def nfe(self):
        nfe = 0
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            layer = getattr(self, layer_name)
            for block in layer:
                nfe += block.nfe
        return nfe / 4

    @nfe.setter
    def nfe(self, value):
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            layer = getattr(self, layer_name)
            for block in layer:
                block.nfe = value

def Get_time_AnodeV2_ResNet18(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], ODEBlock_=ODEBlock, num_classes=num_classes,augment_dim=1)
# 层数之和计算是4（和-4）=目标层数-2
def Get_time_AnodeV2_ResNet34(num_classes):
    return ResNet(BasicBlock, [2, 3, 5, 2], ODEBlock_=ODEBlock, num_classes=num_classes,augment_dim=1)

def Get_time_AnodeV2_ResNet50(num_classes):
    return ResNet(BasicBlock, [3, 4, 6, 3], ODEBlock_=ODEBlock, num_classes=num_classes,augment_dim=1)

def lr_schedule(lr, epoch):
    optim_factor = 0
    if epoch > 250:
        optim_factor = 2
    elif epoch > 150:
        optim_factor = 1
    return lr / math.pow(10, (optim_factor))

