from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class TimmVisionBaseline(nn.Module):
    def __init__(self, model_name: str, num_classes: int, image_size: int = 224):
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise RuntimeError("Install timm to use this vision baseline.") from exc
        self.image_size = image_size
        self.model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != (self.image_size, self.image_size):
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return self.model(x)


def Get_SwinT(num_classes: int):
    return TimmVisionBaseline("swin_tiny_patch4_window7_224", num_classes)


def Get_SwinS(num_classes: int):
    return TimmVisionBaseline("swin_small_patch4_window7_224", num_classes)


def Get_SwinB(num_classes: int):
    return TimmVisionBaseline("swin_base_patch4_window7_224", num_classes)


def Get_ConvNeXtT(num_classes: int):
    return TimmVisionBaseline("convnext_tiny", num_classes)


def Get_ConvNeXtS(num_classes: int):
    return TimmVisionBaseline("convnext_small", num_classes)


def Get_ConvNeXtB(num_classes: int):
    return TimmVisionBaseline("convnext_base", num_classes)


def Get_DeiTTiny(num_classes: int):
    return TimmVisionBaseline("deit_tiny_patch16_224", num_classes)


def Get_DeiTSmall(num_classes: int):
    return TimmVisionBaseline("deit_small_patch16_224", num_classes)


def Get_DeiTBase(num_classes: int):
    return TimmVisionBaseline("deit_base_patch16_224", num_classes)
