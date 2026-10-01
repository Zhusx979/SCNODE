import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint



def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)



class ConcatConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):

        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)



class ODEfunc(nn.Module):
    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConcatConv2d(dim, dim, ksize=3, stride=1, padding=1)
        self.norm2 = norm(dim)
        self.conv2 = ConcatConv2d(dim, dim, ksize=3, stride=1, padding=1)
        self.norm3 = norm(dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        out = self.norm3(out)
        return out



class ODEBlock(nn.Module):
    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time,method='dopri5', rtol=1e-3, atol=1e-3)
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value



class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)



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
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            norm(64),
            nn.ReLU(inplace=True)
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



def Get_odenet(num_classes=1000):
    input_channels = 3
    feature_dim = 64
    model = ODENet(num_classes, feature_dim = 64)
    return model