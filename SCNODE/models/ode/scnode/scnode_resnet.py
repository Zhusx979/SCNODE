import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import abc
from typing import Optional

from torchdiffeq import odeint, odeint_adjoint

from .config import ScnodeConfig

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
    return nn.GroupNorm(math.gcd(32, dim), dim)



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


class FourierFiLMConv2d(nn.Module):
    """Apply a time-conditioned affine modulation to convolution outputs."""

    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super(FourierFiLMConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, *args, **kwargs)
        hidden_channels = max(8, out_channels)
        self.time_affine = nn.Sequential(
            nn.Linear(2, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 2 * out_channels),
        )

    def forward(self, t, x):
        t = t.reshape(1).to(device=x.device, dtype=x.dtype)
        time_features = torch.stack(
            [torch.sin(2 * math.pi * t), torch.cos(2 * math.pi * t)], dim=-1
        )
        gamma, beta = self.time_affine(time_features).chunk(2, dim=-1)
        out = self.conv(x)
        return out * (1 + gamma.view(1, -1, 1, 1)) + beta.view(1, -1, 1, 1)

class ODEBlock(nn.Module):
    def __init__(self, odefunc, Nt=None, method=None, augment_dim=None, config=None):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        if config is None:
            config = ScnodeConfig(
                solver=method if method is not None else "rk4",
                ode_steps=Nt if Nt is not None else 4,
                augment_dim=augment_dim if augment_dim is not None else 1,
            )
        self.config = config
        self.augment_dim = config.augment_dim
        self.options = {'Nt': config.ode_steps, 'method': config.solver}
        self.method = config.solver
        self.solver_options = (
            {'step_size': 1.0 / config.ode_steps}
            if config.solver in {'euler', 'rk4'}
            else None
        )
        integration_time = (
            torch.linspace(0.0, 1.0, config.ode_steps + 1)
            if config.solver in {'euler', 'rk4'}
            else torch.tensor([0.0, 1.0])
        )
        self.register_buffer('integration_time', integration_time)

    def forward(self, x, eval_times=None, zero_auxiliary=False, return_states=False):
        if eval_times is None:
            integration_time = self.integration_time.type_as(x)
        else:
            integration_time = eval_times.type_as(x)


        # 添加 augment_dim 个零通道
        if self.augment_dim > 0:
            batch_size, channels, height, width = x.shape
            aug = torch.zeros(batch_size, self.augment_dim, height, width, device=x.device, dtype=x.dtype)
            x = torch.cat([x, aug], dim=1)  # 通道数变为 dim + augment_dim

        solver_kwargs = {
            'method': self.method,
            'rtol': self.config.rtol,
            'atol': self.config.atol,
        }
        if self.solver_options is not None:
            solver_kwargs['options'] = self.solver_options
        states = odeint(self.odefunc, x, integration_time, **solver_kwargs)
        if return_states:
            return states
        out = states[-1]
        if zero_auxiliary and self.augment_dim > 0:
            main, auxiliary = torch.split(
                out, [out.shape[1] - self.augment_dim, self.augment_dim], dim=1
            )
            out = torch.cat((main, torch.zeros_like(auxiliary)), dim=1)
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

    def __init__(self, dim, num_filters=64,augment_dim=1, time_mode='concat'):
        super(BasicBlock2, self).__init__()
        self.nfe = 0
        self.dim = dim
        self.in_planes = dim
        self.augment_dim = augment_dim
        self.time_mode = time_mode
        self.total_in_planes = dim+self.augment_dim
        if time_mode == 'none':
            self.conv1 = nn.Conv2d(self.total_in_planes, num_filters,
                                   kernel_size=1, stride=1, padding=0)
            self.conv2 = nn.Conv2d(num_filters, num_filters,
                                   kernel_size=3, stride=1, padding=1)
        elif time_mode == 'concat':
            self.conv1 = Conv2dTime(self.total_in_planes, num_filters,
                                    kernel_size=1, stride=1, padding=0)
            self.conv2 = Conv2dTime(num_filters, num_filters,
                                    kernel_size=3, stride=1, padding=1)
        elif time_mode == 'fourier_film':
            self.conv1 = FourierFiLMConv2d(self.total_in_planes, num_filters,
                                           kernel_size=1, stride=1, padding=0)
            self.conv2 = FourierFiLMConv2d(num_filters, num_filters,
                                           kernel_size=3, stride=1, padding=1)
        else:
            raise ValueError('Unsupported time_mode: {}'.format(time_mode))
        # Stateless normalization keeps solver evaluations comparable across time modes.
        self.bn1 = norm(num_filters)
        self.bn2 = norm(num_filters)
        self.shortcut = nn.Sequential()

    def _apply_conv(self, conv, t, x):
        return conv(x) if self.time_mode == 'none' else conv(t, x)

    def _apply_norm(self, norm_layer, t, x):
        return norm_layer(x)

    def forward(self, t, x):
        self.nfe += 1
        # 这里的x已经加上了augment_dim个零通道，为了残差连接，num_filters 必须= dim + augment_dim
        out = self._apply_conv(self.conv1, t, x)
        out = F.relu(self._apply_norm(self.bn1, t, out))
        out = self._apply_conv(self.conv2, t, out)
        out = self._apply_norm(self.bn2, t, out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, ODEBlock_=ODEBlock,
                 augment_dim=None, config=None):
        super(ResNet, self).__init__()
        if config is None:
            config = ScnodeConfig(augment_dim=1 if augment_dim is None else augment_dim)
        elif augment_dim is not None and augment_dim != config.augment_dim:
            raise ValueError('augment_dim must match config.augment_dim when both are provided')
        self.in_planes = 64
        self.ODEBlock = ODEBlock_
        self.config = config
        self.augment_dim = config.augment_dim
        self.ode_entry_size = config.ode_entry_size
        self._first_ode_input_shape = None
        self._last_ode_entry_metadata = {
            'downsampling_applied': None,
            'downsampling_bypassed': None,
            'resize_applied': None,
        }
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv_cifar10 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)#对于cifar10
        self.conv_BM = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)#对于cifar10
        self.bn1 = nn.BatchNorm2d(64)
        # The BM stem maps 224x224 images to 112x112.  Repeating the chosen
        # stride-two operator reaches each requested ODE grid directly, rather
        # than applying one operator followed by a shared adaptive-average pool.
        self.entry_downsample_stages = int(math.log2(112 // config.ode_entry_size))
        if config.downsampling == 'maxpool':
            entry_layers = [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                            for _ in range(self.entry_downsample_stages)]
        elif config.downsampling == 'avgpool':
            entry_layers = [nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
                            for _ in range(self.entry_downsample_stages)]
        else:
            entry_layers = [
                nn.Sequential(
                    nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                )
                for _ in range(self.entry_downsample_stages)
            ]
        self.entry_downsample = nn.Sequential(*entry_layers) if entry_layers else nn.Identity()
        self.maxpool = self.entry_downsample

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

    @property
    def first_ode_input_shape(self):
        """Spatial shape received by the first ODE block during the latest forward path."""
        return self._first_ode_input_shape

    @property
    def ode_entry_metadata(self):
        """Describe the runtime policy and its outcome for the latest ODE entry."""
        return {
            'target_spatial_size': self.ode_entry_size,
            'configured_downsampling': self.config.downsampling,
            'downsampling_policy': 'apply only when it does not undershoot the target',
            'resize_policy': 'adaptive average pooling for downscaling only',
            **self._last_ode_entry_metadata,
        }

    def _can_apply_entry_downsample(self, x):
        """Keep the configured operator only when its stride-two output reaches the target."""
        height, width = x.shape[-2:]
        target = self.ode_entry_size
        return (
            self.entry_downsample_stages > 0
            and height >= target * (2 ** self.entry_downsample_stages)
            and width >= target * (2 ** self.entry_downsample_stages)
        )

    def _prepare_ode_entry(self, x):
        """Apply the configured entry operator and a downscale-only target resize."""
        downsampling_applied = self._can_apply_entry_downsample(x)
        if downsampling_applied:
            x = self.entry_downsample(x)

        height, width = x.shape[-2:]
        resize_applied = (
            height >= self.ode_entry_size
            and width >= self.ode_entry_size
            and (height != self.ode_entry_size or width != self.ode_entry_size)
        )
        if resize_applied:
            x = F.adaptive_avg_pool2d(
                x, (self.ode_entry_size, self.ode_entry_size)
            )

        self._last_ode_entry_metadata = {
            'downsampling_applied': downsampling_applied,
            'downsampling_bypassed': not downsampling_applied,
            'resize_applied': resize_applied,
        }
        return x

    def forward_to_first_ode_input(self, x):
        """Run the stem through the residual block immediately preceding the first ODE."""
        out = F.relu(self.bn1(self.conv_BM(x)))
        out = self._prepare_ode_entry(out)
        out = self.layer1[0](out)
        self._first_ode_input_shape = tuple(out.shape[-2:])
        return out

    def _make_layer(self, out_planes, num_blocks, stride):
        layers = []
        strides = [stride] + [1] * (num_blocks - 1)
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, out_planes, stride))
            self.in_planes = out_planes * BasicBlock.expansion
            current_out_planes = out_planes + self.augment_dim #决定了num_filters
            block = BasicBlock2(self.in_planes, num_filters=current_out_planes,
                                augment_dim=self.augment_dim, time_mode=self.config.time_mode)
            if self.ODEBlock is ODEBlock:
                layers.append(self.ODEBlock(block, config=self.config))
            else:
                layers.append(self.ODEBlock(block, augment_dim=self.augment_dim))
            self.in_planes = current_out_planes
        return nn.Sequential(*layers)

    def _forward_layers(
        self, layers, x, zero_auxiliary=False, terminal_ode_block=None
    ):
        for layer in layers:
            if isinstance(layer, ODEBlock):
                x = layer(
                    x,
                    zero_auxiliary=zero_auxiliary and layer is terminal_ode_block,
                )
            else:
                x = layer(x)
        return x

    def forward_with_trajectory(self, x, time_points):
        """Return logits and per-ODE-block computational trajectories.

        These are solver states inside one trained classifier, not longitudinal
        observations of a biological cell.
        """
        out = self.forward_to_first_ode_input(x)
        trajectories = {}
        stages = (("layer1", self.layer1[1:]), ("layer2", self.layer2),
                  ("layer3", self.layer3), ("layer4", self.layer4))
        for stage_name, stage in stages:
            for index, layer in enumerate(stage):
                if isinstance(layer, ODEBlock):
                    states = layer(out, eval_times=time_points, return_states=True)
                    trajectories[f"{stage_name}_{index}"] = states
                    out = states[-1]
                else:
                    out = layer(out)
        out = self.pool(out).view(out.size(0), -1)
        return self.linear(out), trajectories

    def forward(self, x, zero_auxiliary=False):
        out = self.forward_to_first_ode_input(x)
        stages = (self.layer1[1:], self.layer2, self.layer3, self.layer4)
        terminal_ode_block = next(
            (
                layer
                for stage in reversed(stages)
                for layer in reversed(stage)
                if isinstance(layer, ODEBlock)
            ),
            None,
        )
        for stage in stages:
            out = self._forward_layers(
                stage,
                out,
                zero_auxiliary=zero_auxiliary,
                terminal_ode_block=terminal_ode_block,
            )
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

def Get_time_AnodeV2_ResNet18(num_classes, config: Optional[ScnodeConfig] = None):
    return ResNet(BasicBlock, [2, 2, 2, 2], ODEBlock_=ODEBlock, num_classes=num_classes,
                  config=config)
# 层数之和计算是4（和-4）=目标层数-2
def Get_time_AnodeV2_ResNet34(num_classes, config: Optional[ScnodeConfig] = None):
    return ResNet(BasicBlock, [2, 3, 5, 2], ODEBlock_=ODEBlock, num_classes=num_classes,
                  config=config)

def Get_time_AnodeV2_ResNet50(num_classes, config: Optional[ScnodeConfig] = None):
    return ResNet(BasicBlock, [3, 4, 6, 3], ODEBlock_=ODEBlock, num_classes=num_classes,
                  config=config)

def lr_schedule(lr, epoch):
    optim_factor = 0
    if epoch > 250:
        optim_factor = 2
    elif epoch > 150:
        optim_factor = 1
    return lr / math.pow(10, (optim_factor))

