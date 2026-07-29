import os
import argparse
import logging
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

parser = argparse.ArgumentParser()
parser.add_argument('--network', type=str, choices=['resnet', 'odenet'], default='resnet')
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--adjoint', type=eval, default=False, choices=[True, False])
parser.add_argument('--downsampling-method', type=str, default='conv', choices=['conv', 'res'])
parser.add_argument('--nepochs', type=int, default=20)
parser.add_argument('--data_aug', type=eval, default=True, choices=[True, False])
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--test_batch_size', type=int, default=128)
parser.add_argument('--image_size', type=int, default=224)  # 新增参数，适应你的数据尺寸
parser.add_argument('--save', type=str, default='./experiment1')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, stride=1, padding=1, dilation=1, bias=True):
        super(DepthwiseSeparableConv2d, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        self.depthwise = nn.Conv2d(
            dim_in, dim_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=dim_in, bias=False
        )
        self.norm = nn.BatchNorm2d(dim_in)
        self.relu = nn.ReLU(inplace=False)
        self.pointwise = nn.Conv2d(
            dim_in, dim_out, kernel_size=1, stride=1, padding=0, bias=bias
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.pointwise(x)
        return x

class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.norm1 = norm(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm2 = norm(planes)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x):
        shortcut = x
        out = self.relu(self.norm1(x))
        if self.downsample is not None:
            shortcut = self.downsample(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + shortcut


class ConcatConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, stride=1, padding=0, dilation=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else DepthwiseSeparableConv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=kernel_size, stride=stride, padding=padding,
            dilation=dilation, bias=bias
        )

    def forward(self, t, x):
        assert torch.isfinite(t), "Time variable t must be finite"
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)

# ODE 动态函数
import torch
import torch.nn as nn

class ODEfunc(nn.Module):
    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        # 归一化层
        self.norm1 = norm(dim)
        self.norm2 = norm(dim)
        self.norm3 = norm(dim)
        self.norm4 = nn.LayerNorm([dim, 7, 7])  # 适配 7x7 特征图

        # 激活函数
        self.relu = nn.ReLU(inplace=False)

        # 卷积层
        self.conv1 = ConcatConv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        # 多尺度卷积：并行 3x3 和 5x5
        self.conv2_3x3 = ConcatConv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1)
        self.conv2_5x5 = ConcatConv2d(dim, dim // 2, kernel_size=5, stride=1, padding=2)
        self.conv3 = ConcatConv2d(dim, dim, kernel_size=3, stride=1, padding=1)

        # 通道注意力 (Squeeze-and-Excitation)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 16, kernel_size=1, bias=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(dim // 16, dim, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1

        # 第一层：归一化 + 激活 + 卷积 + 残差
        out = self.norm1(x)
        out = self.relu(out)
        residual = out
        out = self.conv1(t, out)
        out = out + residual

        # 第二层：多尺度卷积 + 通道注意力
        out = self.norm2(out)
        out = self.relu(out)
        residual = out
        out_3x3 = self.conv2_3x3(t, out)
        out_5x5 = self.conv2_5x5(t, out)
        out = torch.cat([out_3x3, out_5x5], dim=1)
        se_weight = self.se(out)
        out = out * se_weight
        out = out + residual

        # 第三层：归一化 + 激活 + 卷积 + 残差
        out = self.norm3(out)
        out = self.relu(out)
        residual = out
        out = self.conv3(t, out)
        out = out + residual

        # 最后一层：LayerNorm
        out = self.norm4(out)

        return out


class ODEBlock(nn.Module):
    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=args.tol, atol=args.tol)
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


class RunningAverageMeter(object):
    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def get_custom_loaders(data_aug=False, batch_size=64, test_batch_size=64):
    data_dir = "../../archive/datasets/all_dataset/300_BM_dataset"

    if data_aug:
        transform_train = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.RandomRotation(5),
            transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        transform_train = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    transform_test = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform_train)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform_test)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    train_eval_loader = DataLoader(train_dataset, batch_size=test_batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader, train_eval_loader, val_loader


def inf_generator(iterable):
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def learning_rate_with_decay(batch_size, batch_denom, batches_per_epoch, boundary_epochs, decay_rates):
    initial_learning_rate = args.lr * batch_size / batch_denom
    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]

    def learning_rate_fn(itr):
        lt = [itr < b for b in boundaries] + [True]
        i = np.argmax(lt)
        return vals[i]

    return learning_rate_fn


def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)


def accuracy(model, dataset_loader):
    total_correct = 0
    for x, y in dataset_loader:
        x = x.to(device)
        y = one_hot(np.array(y.numpy()), len(dataset_loader.dataset.classes))
        target_class = np.argmax(y, axis=1)
        predicted_class = np.argmax(model(x).cpu().detach().numpy(), axis=1)
        total_correct += np.sum(predicted_class == target_class)
    return total_correct / len(dataset_loader.dataset)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, 'r', encoding='gbk', errors='ignore') as f:
        logger.info(f.read())
    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())
    return logger

if __name__ == '__main__':
    makedirs(args.save)
    logger = get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
    logger.info(args)

    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    is_odenet = args.network == 'odenet'

    # 加载数据集获取类别数
    train_loader, _, _, _ = get_custom_loaders()
    num_classes = len(train_loader.dataset.classes)

    in_channels = 3
    feature_dim = 64
    # 修改模型结构适应RGB输入和你的类别数
    if args.downsampling_method == 'conv':
        downsampling_layers = [
            DepthwiseSeparableConv2d(in_channels, feature_dim, kernel_size=7, stride=2, padding=3, bias=False),
            DepthwiseSeparableConv2d(feature_dim, feature_dim, kernel_size=3, stride=2, padding=1, bias=False),
            DepthwiseSeparableConv2d(feature_dim, feature_dim, kernel_size=3, stride=2, padding=1, bias=False),
            DepthwiseSeparableConv2d(feature_dim, feature_dim, kernel_size=3, stride=2, padding=1, bias=False),
            DepthwiseSeparableConv2d(feature_dim, feature_dim, kernel_size=3, stride=2, padding=1, bias=False),

        ]
    elif args.downsampling_method == 'res':
        downsampling_layers = [
            nn.Conv2d(3, 64, 3, 1),  # 输入通道改为3
            ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2)),
            ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2)),
        ]


    feature_layers = [ODEBlock(ODEfunc(64))] if is_odenet else [ResBlock(64, 64) for _ in range(6)]
    fc_layers = [
        norm(64),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1, 1)),
        Flatten(),
        nn.Linear(64, num_classes)  # 输出改为你的类别数
    ]
    from SCNODE.models.baselines.efficientnet_baseline import Get_efficientnet
    model = nn.Sequential(*downsampling_layers, *feature_layers, *fc_layers).to(device)
    # model = Get_efficientnet(num_classes).to(device)
    logger.info(model)

    logger.info(f'Number of parameters: {count_parameters(model)}')
    logger.info(f'Number of classes: {num_classes}')

    criterion = nn.CrossEntropyLoss().to(device)
    train_loader, test_loader, train_eval_loader, val_loader = get_custom_loaders(
        args.data_aug, args.batch_size, args.test_batch_size
    )

    data_gen = inf_generator(train_loader)
    batches_per_epoch = len(train_loader)

    lr_fn = learning_rate_with_decay(
        args.batch_size, batch_denom=64, batches_per_epoch=batches_per_epoch,
        boundary_epochs=[60, 100, 140], decay_rates=[1, 0.1, 0.01, 0.001]
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    best_acc = 0
    batch_time_meter = RunningAverageMeter()
    f_nfe_meter = RunningAverageMeter()
    b_nfe_meter = RunningAverageMeter()
    end = time.time()

    for itr in range(args.nepochs * batches_per_epoch):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_fn(itr)

        optimizer.zero_grad()
        x, y = data_gen.__next__()
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        if is_odenet:
            nfe_forward = feature_layers[0].nfe
            feature_layers[0].nfe = 0

        loss.backward()
        optimizer.step()

        if is_odenet:
            nfe_backward = feature_layers[0].nfe
            feature_layers[0].nfe = 0

        batch_time_meter.update(time.time() - end)
        if is_odenet:
            f_nfe_meter.update(nfe_forward)
            b_nfe_meter.update(nfe_backward)
        end = time.time()

        if itr % batches_per_epoch == 0:
            with torch.no_grad():
                train_acc = accuracy(model, train_eval_loader)
                val_acc = accuracy(model, val_loader)
                test_acc = accuracy(model, test_loader)
                if val_acc > best_acc:
                    torch.save({'state_dict': model.state_dict(), 'args': args}, os.path.join(args.save, 'model.pth'))
                    best_acc = val_acc
                logger.info(
                    "Epoch {:04d} | Time {:.3f} ({:.3f}) | NFE-F {:.1f} | NFE-B {:.1f} | "
                    "Train Acc {:.4f} | Val Acc {:.4f} | Test Acc {:.4f}".format(
                        itr // batches_per_epoch, batch_time_meter.val, batch_time_meter.avg,
                        f_nfe_meter.avg, b_nfe_meter.avg, train_acc, val_acc, test_acc
                    )
                )
