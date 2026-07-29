import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from SCNODE.training.classification_trainer import conv_init, train_val_test_model
from SCNODE.training.experiment_config import args, models

# 数据集存储路径
data_dir = str(ROOT_DIR / "SCNODE" / "archive" / "datasets" / "cifar10_data")

def _import_torchvision():
    try:
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise ImportError(
            "run_cifar10_experiment.py requires torchvision at runtime. "
            "Install torchvision before training or evaluating CIFAR-10."
        ) from exc
    return datasets, transforms

def build_cifar10_dataloaders():
    datasets, transforms = _import_torchvision()
    transform1 = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
    ])

    transform2 = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
    ])

    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform1)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform2)
    train_size = 50000
    val_size = 0
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    testloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=0)
    return train_dataset, val_dataset, test_dataset, trainloader, valloader, testloader

# 设置随机种子以确保实验可复现
def set_seed(seed=42):
    random.seed(seed)  # Python 内置随机数种子
    np.random.seed(seed)  # NumPy 随机种子
    torch.manual_seed(seed)  # PyTorch CPU 随机种子
    torch.cuda.manual_seed(seed)  # PyTorch GPU 随机种子
    torch.cuda.manual_seed_all(seed)  # 多 GPU 随机种子
    torch.backends.cudnn.deterministic = True  # 确保 CUDA 计算可复现
    torch.backends.cudnn.benchmark = False  # 关闭自动优化以确保一致性
def main():
    set_seed(42)
    train_dataset, val_dataset, test_dataset, trainloader, valloader, testloader = build_cifar10_dataloaders()

    print(f'Train dataset size: {len(train_dataset)}')
    print(f'Val dataset size: {len(val_dataset)}')
    print(f'Test dataset size: {len(test_dataset)}')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 10
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    print(f"Num_class: {num_classes}, Class: {class_names}")

    for i in range(46, 47):
        set_seed(i)
        for model_spec, name in models:
            print(f"Train with seed {i} ")
            model_func = model_spec.load_factory()
            model = model_func(num_classes=num_classes).to(device)
            model.apply(conv_init)
            criterion = torch.nn.CrossEntropyLoss()
            print(f"Training model: {name}")
            train_val_test_model(
                model,
                trainloader,
                None,
                testloader,
                criterion,
                device,
                name,
                class_names,
                num_epochs=args.num_epochs
            )
            print(f"完成模型 {name} 的训练")


if __name__ == "__main__":
    main()
