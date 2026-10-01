import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import abc
from dataclasses import replace
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

    def __init__(self, num_features, window_size=5, num_grids=11, eps=1e-5, momentum=0.1, T=1.0, affine=True, grid_policy="uniform"):
        super(TWBN, self).__init__()
        if window_size < 1 or num_grids < 2:
            raise ValueError("window_size must be positive and num_grids must be at least two")
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.T = T
        self.affine = affine
        self.window_size = min(window_size, num_grids)
        self.num_grids = num_grids
        grid = torch.linspace(0, 1, num_grids).pow(0.5) * T if grid_policy == "quantile" else torch.linspace(0, T, num_grids)
        self.register_buffer('T_star', grid)
        self.register_buffer('running_mean', torch.zeros(num_grids, num_features))
        self.register_buffer('running_var', torch.ones(num_grids, num_features))

        if affine:
            self.gamma = nn.Parameter(torch.ones(num_grids, num_features))
            self.beta = nn.Parameter(torch.zeros(num_grids, num_features))
        else:
            self.register_buffer('gamma', torch.ones(num_grids, num_features))
            self.register_buffer('beta', torch.zeros(num_grids, num_features))

    def _find_window(self, t, T_star):
        """Find the window of time grids based on the time t."""
        l = torch.searchsorted(T_star, t.detach()).item() - 1
        l = max(0, min(l, self.num_grids - 2))
        return l

    def _smooth_stats(self, stats, l):
        """Apply smoothing to the stats over the sliding window."""

        end = min(l + self.window_size, self.num_grids)
        offsets = torch.arange(end - l, device=stats.device, dtype=stats.dtype)
        weights = torch.clamp(1.0 - offsets / self.window_size, min=0.1)
        return torch.sum(stats[l:end] * weights.view(-1, 1), dim=0) / weights.sum()

    def _interpolate(self, values, omega1, omega2, l):
        """Perform linear interpolation between l and l+1."""
        return omega1 * values[l] + omega2 * values[l + 1]

    def forward(self, x, t, training=None):
        """
        Forward pass with time window-based batch normalization.

        Args:
            x (torch.Tensor): Input (B, C, H, W)
            t (float): Current time tj
            training (bool): Mode flag

        Returns:
            torch.Tensor: Normalized x
        """
        if x.ndim != 4 or x.shape[1] != self.num_features:
            raise ValueError("TWBN expects a BCHW tensor with the configured channel count")
        use_batch_stats = self.training if training is None else training
        t = t.to(device=x.device, dtype=x.dtype).reshape(())
        grid = self.T_star.to(device=x.device, dtype=x.dtype)
        l = self._find_window(t, grid)
        upper = l + 1
        omega2 = ((t - grid[l]) / (grid[upper] - grid[l])).clamp(0.0, 1.0)
        omega1 = 1.0 - omega2
        gamma_j = self._interpolate(self.gamma, omega1, omega2, l)
        beta_j = self._interpolate(self.beta, omega1, omega2, l)

        if use_batch_stats:
            batch_mean = x.mean(dim=(0, 2, 3))
            batch_var = x.var(dim=(0, 2, 3), unbiased=False)
            historical_mean = self.running_mean.to(x).clone()
            historical_var = self.running_var.to(x).clone()
            historical_mean[l] = batch_mean
            historical_var[l] = batch_var
            mean = self._smooth_stats(historical_mean, l)
            var = self._smooth_stats(historical_var, l)
            with torch.no_grad():
                self.running_mean[l].lerp_(mean.detach(), self.momentum * float(omega1))
                self.running_var[l].lerp_(var.detach(), self.momentum * float(omega1))
        else:
            mean = self._interpolate(self.running_mean, omega1, omega2, l).to(x)
            var = self._interpolate(self.running_var, omega1, omega2, l).to(x)

        normalized = (x - mean.view(1, -1, 1, 1)) / torch.sqrt(var.view(1, -1, 1, 1) + self.eps)
        return normalized * gamma_j.view(1, -1, 1, 1) + beta_j.view(1, -1, 1, 1)


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



        if self.augment_dim > 0:
            batch_size, channels, height, width = x.shape
            aug = torch.zeros(batch_size, self.augment_dim, height, width, device=x.device, dtype=x.dtype)
            x = torch.cat([x, aug], dim=1)

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

    def __init__(self, in_planes, planes, stride=1, use_bn=True):
        super(BasicBlock, self).__init__()
        self.nfe = 0
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes) if use_bn else nn.Identity()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes) if use_bn else nn.Identity()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes) if use_bn else nn.Identity()
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BasicBlock2(nn.Module):
    expansion = 1

    def __init__(self, dim, num_filters=64, augment_dim=1, time_mode='concat', use_bn=True, use_twbn=True,
                 twbn_window=5, twbn_grids=11, grid_policy="uniform"):
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
        if use_twbn:
            self.bn1 = TWBN(num_filters, window_size=twbn_window, num_grids=twbn_grids, grid_policy=grid_policy)
            self.bn2 = TWBN(num_filters, window_size=twbn_window, num_grids=twbn_grids, grid_policy=grid_policy)
        elif use_bn:
            self.bn1 = nn.BatchNorm2d(num_filters)
            self.bn2 = nn.BatchNorm2d(num_filters)
        else:
            self.bn1 = nn.Identity()
            self.bn2 = nn.Identity()
        self.shortcut = nn.Sequential()

    def _apply_conv(self, conv, t, x):
        return conv(x) if self.time_mode == 'none' else conv(t, x)

    def _apply_norm(self, norm_layer, t, x):
        return norm_layer(x, t) if isinstance(norm_layer, TWBN) else norm_layer(x)

    def forward(self, t, x):
        self.nfe += 1

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
        if not config.use_sam and config.augment_dim != 0:
            config = replace(config, augment_dim=0)
        if not config.use_tconv and config.time_mode != "none":
            config = replace(config, time_mode="none")
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
        self.conv_cifar10 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_BM = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64) if config.use_bn else nn.Identity()



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
                    nn.BatchNorm2d(64) if config.use_bn else nn.Identity(),
                )
                for _ in range(self.entry_downsample_stages)
            ]
        self.entry_downsample = nn.Sequential(*entry_layers) if entry_layers else nn.Identity()
        self.maxpool = self.entry_downsample


        self.layer1 = self._make_layer(out_planes=64, num_blocks=num_blocks[0] - 1, stride=1)

        self.layer2 = self._make_layer(out_planes=128, num_blocks=num_blocks[1] - 1, stride=2)

        self.layer3 = self._make_layer(out_planes=256, num_blocks=num_blocks[2] - 1, stride=2)

        self.layer4 = self._make_layer(out_planes=512, num_blocks=num_blocks[3] - 1, stride=2)

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
        stem = self.conv_cifar10 if x.shape[-2:] == (32, 32) else self.conv_BM
        out = F.relu(self.bn1(stem(x)))
        out = self._prepare_ode_entry(out)
        out = self.layer1[0](out)
        self._first_ode_input_shape = tuple(out.shape[-2:])
        return out

    def _make_layer(self, out_planes, num_blocks, stride):
        layers = []
        strides = [stride] + [1] * (num_blocks - 1)
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, out_planes, stride, use_bn=self.config.use_bn))
            self.in_planes = out_planes * BasicBlock.expansion
            current_out_planes = out_planes + self.augment_dim
            block = BasicBlock2(self.in_planes, num_filters=current_out_planes,
                                augment_dim=self.augment_dim, time_mode=self.config.time_mode,
                                use_bn=self.config.use_bn, use_twbn=self.config.use_twbn,
                                twbn_window=self.config.twbn_window, twbn_grids=self.config.twbn_grids,
                                grid_policy=self.config.grid_policy)
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
            if type(layer) is ODEBlock:
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

