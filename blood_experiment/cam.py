from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def _normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    array = np.asarray(heatmap, dtype=float)
    array = np.maximum(array, 0.0)
    max_value = float(array.max()) if array.size else 0.0
    if max_value == 0.0:
        return np.zeros_like(array)
    return array / max_value


def save_overlay_preview(
    output_path: Path | str,
    image: np.ndarray,
    heatmap: np.ndarray,
    title: str,
) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    base_image = np.asarray(image)
    if base_image.dtype != np.uint8:
        base_image = np.clip(base_image, 0, 255).astype(np.uint8)

    normalized_heatmap = _normalize_heatmap(heatmap)
    fig, ax = plt.subplots(figsize=(5, 5), dpi=180)
    ax.imshow(base_image)
    ax.imshow(normalized_heatmap, cmap="jet", alpha=0.42)
    ax.set_title(title, fontsize=12, pad=10)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(destination, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return destination


@dataclass
class CamArtifact:
    image_path: str
    predicted_label: str
    true_label: str
    gradcam_path: Path
    smoothgradcam_path: Path


def save_cam_bundle(
    output_dir: Path | str,
    overlays: list[dict[str, Any]],
) -> list[CamArtifact]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    artifacts: list[CamArtifact] = []

    for index, overlay in enumerate(overlays):
        gradcam_path = save_overlay_preview(
            destination / f"sample_{index:03d}_gradcam.png",
            overlay["image"],
            overlay["gradcam_heatmap"],
            overlay.get("gradcam_title", "Grad-CAM"),
        )
        smoothgradcam_path = save_overlay_preview(
            destination / f"sample_{index:03d}_smoothgradcam.png",
            overlay["image"],
            overlay["smoothgradcam_heatmap"],
            overlay.get("smoothgradcam_title", "Smooth Grad-CAM"),
        )
        artifacts.append(
            CamArtifact(
                image_path=str(overlay.get("image_path", "")),
                predicted_label=str(overlay.get("predicted_label", "")),
                true_label=str(overlay.get("true_label", "")),
                gradcam_path=gradcam_path,
                smoothgradcam_path=smoothgradcam_path,
            )
        )
    return artifacts


class _CamHook:
    def __init__(self, module: torch.nn.Module) -> None:
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.forward_handle = module.register_forward_hook(self._capture_activations)
        self.backward_handle = module.register_full_backward_hook(self._capture_gradients)

    def _capture_activations(self, module, inputs, output) -> None:
        self.activations = output.detach()

    def _capture_gradients(self, module, grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def close(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()


def find_last_conv_layer(model: torch.nn.Module) -> tuple[str, torch.nn.Module]:
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, torch.nn.Conv2d):
            return name, module
    raise ValueError("No Conv2d layer found for Grad-CAM generation.")


def resolve_target_layer(
    model: torch.nn.Module,
    layer_name: str | None = None,
) -> tuple[str, torch.nn.Module]:
    if layer_name is None:
        return find_last_conv_layer(model)
    modules = dict(model.named_modules())
    if layer_name not in modules:
        raise ValueError(f"Target layer '{layer_name}' was not found in the model.")
    return layer_name, modules[layer_name]


def _compute_cam_heatmap(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    class_index: int,
    target_layer: torch.nn.Module,
) -> np.ndarray:
    hook = _CamHook(target_layer)
    try:
        model.zero_grad(set_to_none=True)
        outputs = model(input_tensor.unsqueeze(0))
        target_score = outputs[:, class_index].sum()
        target_score.backward()

        if hook.activations is None or hook.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations or gradients.")

        weights = hook.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * hook.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[1:],
            mode="bilinear",
            align_corners=False,
        )
        heatmap = cam[0, 0].detach().cpu().numpy()
        return _normalize_heatmap(heatmap)
    finally:
        hook.close()


def _tensor_to_uint8_image(
    tensor: torch.Tensor,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> np.ndarray:
    image = tensor.detach().cpu().permute(1, 2, 0).numpy()
    image = image * np.asarray(std).reshape(1, 1, 3) + np.asarray(mean).reshape(1, 1, 3)
    image = np.clip(image, 0.0, 1.0)
    return (image * 255).astype(np.uint8)


def generate_gradcam_heatmap(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    class_index: int,
    target_layer_name: str | None = None,
) -> np.ndarray:
    _, target_layer = resolve_target_layer(model, target_layer_name)
    return _compute_cam_heatmap(model, input_tensor, class_index, target_layer)


def generate_smoothgradcam_heatmap(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    class_index: int,
    target_layer_name: str | None = None,
    noise_samples: int = 8,
    noise_sigma: float = 0.1,
) -> np.ndarray:
    _, target_layer = resolve_target_layer(model, target_layer_name)
    heatmaps = []
    for _ in range(max(noise_samples, 1)):
        noise = torch.randn_like(input_tensor) * noise_sigma
        noisy_input = input_tensor + noise
        heatmaps.append(_compute_cam_heatmap(model, noisy_input, class_index, target_layer))
    return _normalize_heatmap(np.mean(np.stack(heatmaps, axis=0), axis=0))


def generate_cam_overlays(
    model: torch.nn.Module,
    samples: list[dict[str, Any]],
    class_names: list[str],
    output_dir: Path | str,
    target_layer_name: str | None = None,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    noise_samples: int = 8,
    noise_sigma: float = 0.1,
) -> list[CamArtifact]:
    base_model = model.module if hasattr(model, "module") else model
    base_model.eval()
    overlays: list[dict[str, Any]] = []

    for sample in samples:
        tensor = sample["input_tensor"]
        if tensor.device.type != "cpu":
            tensor = tensor.detach().cpu()
        predicted_index = int(sample["predicted_index"])
        image = _tensor_to_uint8_image(tensor, mean=mean, std=std)
        gradcam_heatmap = generate_gradcam_heatmap(
            base_model,
            tensor,
            predicted_index,
            target_layer_name=target_layer_name,
        )
        smoothgradcam_heatmap = generate_smoothgradcam_heatmap(
            base_model,
            tensor,
            predicted_index,
            target_layer_name=target_layer_name,
            noise_samples=noise_samples,
            noise_sigma=noise_sigma,
        )
        overlays.append(
            {
                "image": image,
                "image_path": sample["image_path"],
                "predicted_label": class_names[predicted_index],
                "true_label": class_names[int(sample["true_index"])],
                "gradcam_heatmap": gradcam_heatmap,
                "smoothgradcam_heatmap": smoothgradcam_heatmap,
                "gradcam_title": f"Grad-CAM | Pred: {class_names[predicted_index]}",
                "smoothgradcam_title": f"Smooth Grad-CAM | Pred: {class_names[predicted_index]}",
            }
        )

    return save_cam_bundle(output_dir=output_dir, overlays=overlays)
