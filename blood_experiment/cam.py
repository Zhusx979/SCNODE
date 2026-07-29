from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from blood_experiment.plot_style import CAM_CMAP, finalize_figure


def _normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    array = np.asarray(heatmap, dtype=float)
    array = np.maximum(array, 0.0)
    max_value = float(array.max()) if array.size else 0.0
    if max_value == 0.0:
        return np.zeros_like(array)
    return array / max_value


def _prepare_base_image(image: np.ndarray) -> np.ndarray:
    base_image = np.asarray(image)
    if base_image.dtype != np.uint8:
        base_image = np.clip(base_image, 0, 255).astype(np.uint8)
    return base_image


def _prepare_grayscale_image(image: np.ndarray) -> np.ndarray:
    base_image = _prepare_base_image(image).astype(float)
    grayscale = np.dot(base_image[..., :3], np.array([0.299, 0.587, 0.114], dtype=float))
    return np.clip(grayscale, 0.0, 255.0).astype(np.uint8)


def _activation_alpha(normalized_heatmap: np.ndarray) -> np.ndarray:
    return np.clip(np.power(normalized_heatmap, 0.8), 0.14, 0.82)


def _smoothgrad_alpha(normalized_heatmap: np.ndarray) -> np.ndarray:
    return np.clip(np.power(normalized_heatmap, 1.2), 0.08, 0.95)


def _draw_contours(axis, heatmap: np.ndarray) -> None:
    finite_values = np.unique(np.asarray(heatmap, dtype=float))
    contour_levels = [level for level in (0.35, 0.55, 0.75) if finite_values.min() < level < finite_values.max()]
    if not contour_levels:
        return
    axis.contour(
        heatmap,
        levels=contour_levels,
        colors=["#FFF7D6", "#F4A261", "#9A3412"][: len(contour_levels)],
        linewidths=[0.5, 0.65, 0.9][: len(contour_levels)],
        alpha=0.65,
    )


def _draw_cam_panel(axis, base_image: np.ndarray, heatmap: np.ndarray | None, title: str) -> Any:
    axis.imshow(base_image)
    overlay_artist = None
    if heatmap is not None:
        overlay_artist = axis.imshow(
            heatmap,
            cmap=CAM_CMAP,
            alpha=_activation_alpha(heatmap),
        )
        _draw_contours(axis, heatmap)
    axis.set_title(title, fontsize=11.8, pad=10, loc="left")
    axis.axis("off")
    return overlay_artist


def _draw_smoothgrad_panel(axis, grayscale_image: np.ndarray, heatmap: np.ndarray | None, title: str) -> Any:
    axis.set_facecolor("black")
    axis.imshow(grayscale_image, cmap="gray", vmin=0, vmax=255)
    overlay_artist = None
    if heatmap is not None:
        overlay_artist = axis.imshow(
            heatmap,
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            alpha=_smoothgrad_alpha(heatmap),
        )
    axis.set_title(title, fontsize=11.8, pad=10, loc="left")
    axis.axis("off")
    return overlay_artist


def _draw_cam_panel_clean(axis, base_image: np.ndarray, heatmap: np.ndarray | None) -> None:
    axis.imshow(base_image)
    if heatmap is not None:
        axis.imshow(
            heatmap,
            cmap=CAM_CMAP,
            alpha=_activation_alpha(heatmap),
        )
        _draw_contours(axis, heatmap)
    axis.axis("off")


def _draw_smoothgrad_panel_clean(axis, grayscale_image: np.ndarray, heatmap: np.ndarray | None) -> None:
    axis.set_facecolor("black")
    axis.imshow(grayscale_image, cmap="gray", vmin=0, vmax=255)
    if heatmap is not None:
        axis.imshow(
            heatmap,
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            alpha=_smoothgrad_alpha(heatmap),
        )
    axis.axis("off")


def _sanitize_filename(label: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_-]+", "_", label.strip())
    sanitized = sanitized.strip("_")
    return sanitized or "cell"


def save_overlay_preview(
    output_path: Path | str,
    image: np.ndarray,
    heatmap: np.ndarray,
    title: str,
    mode: str = "gradcam",
) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    base_image = _prepare_base_image(image)
    normalized_heatmap = _normalize_heatmap(heatmap)

    fig, ax = plt.subplots(figsize=(4.9, 4.9), dpi=220)
    if mode == "smoothgrad":
        overlay_artist = _draw_smoothgrad_panel(
            ax,
            _prepare_grayscale_image(base_image),
            normalized_heatmap,
            title,
        )
    else:
        overlay_artist = _draw_cam_panel(ax, base_image, normalized_heatmap, title)
    colorbar = fig.colorbar(overlay_artist, ax=ax, fraction=0.045, pad=0.03, shrink=0.88)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(labelsize=8.5)
    colorbar.set_label("Saliency" if mode == "smoothgrad" else "Activation", fontsize=9)
    fig.tight_layout()
    saved_path = finalize_figure(fig, destination)
    plt.close(fig)
    return saved_path


def save_cam_comparison_figure(
    output_path: Path | str,
    image: np.ndarray,
    gradcam_heatmap: np.ndarray,
    smoothgrad_heatmap: np.ndarray,
) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    base_image = _prepare_base_image(image)
    grayscale_image = _prepare_grayscale_image(base_image)
    normalized_gradcam = _normalize_heatmap(gradcam_heatmap)
    normalized_smooth = _normalize_heatmap(smoothgrad_heatmap)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(4.2, 7.8),
        dpi=220,
        gridspec_kw={"hspace": 0.02},
        constrained_layout=True,
    )
    _draw_cam_panel_clean(axes[0], base_image, normalized_gradcam)
    _draw_smoothgrad_panel_clean(axes[1], grayscale_image, normalized_smooth)
    saved_path = finalize_figure(fig, destination)
    plt.close(fig)
    return saved_path


@dataclass
class CamArtifact:
    image_path: str
    predicted_label: str
    true_label: str
    comparison_path: Path


def save_cam_bundle(
    output_dir: Path | str,
    overlays: list[dict[str, Any]],
) -> list[CamArtifact]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    artifacts: list[CamArtifact] = []
    name_counters: dict[str, int] = {}

    for overlay in overlays:
        base_name = _sanitize_filename(str(overlay.get("true_label", "")))
        name_counters[base_name] = name_counters.get(base_name, 0) + 1
        file_stem = base_name if name_counters[base_name] == 1 else f"{base_name}_{name_counters[base_name]:02d}"
        comparison_path = save_cam_comparison_figure(
            destination / f"{file_stem}.png",
            image=overlay["image"],
            gradcam_heatmap=overlay["gradcam_heatmap"],
            smoothgrad_heatmap=overlay["smoothgrad_heatmap"],
        )
        artifacts.append(
            CamArtifact(
                image_path=str(overlay.get("image_path", "")),
                predicted_label=str(overlay.get("predicted_label", "")),
                true_label=str(overlay.get("true_label", "")),
                comparison_path=comparison_path,
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


def infer_model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _compute_cam_heatmap(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    class_index: int,
    target_layer: torch.nn.Module,
) -> np.ndarray:
    hook = _CamHook(target_layer)
    try:
        model_device = infer_model_device(model)
        prepared_input = input_tensor.unsqueeze(0).to(model_device)
        model.zero_grad(set_to_none=True)
        outputs = model(prepared_input)
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


def _compute_input_gradient(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    class_index: int,
) -> torch.Tensor:
    model_device = infer_model_device(model)
    prepared_input = input_tensor.unsqueeze(0).to(model_device).detach().clone()
    prepared_input.requires_grad_(True)
    model.zero_grad(set_to_none=True)
    outputs = model(prepared_input)
    target_score = outputs[:, class_index].sum()
    target_score.backward()
    if prepared_input.grad is None:
        raise RuntimeError("SmoothGrad did not capture input gradients.")
    return prepared_input.grad.detach()[0]


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


def generate_smoothgrad_heatmap(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    class_index: int,
    target_layer_name: str | None = None,
    noise_samples: int = 8,
    noise_sigma: float = 0.1,
) -> np.ndarray:
    del target_layer_name
    gradients = []
    for _ in range(max(noise_samples, 1)):
        noise = torch.randn_like(input_tensor) * noise_sigma
        noisy_input = input_tensor + noise
        gradients.append(_compute_input_gradient(model, noisy_input, class_index).abs())
    mean_gradient = torch.stack(gradients, dim=0).mean(dim=0)
    heatmap = mean_gradient.mean(dim=0).detach().cpu().numpy()
    return _normalize_heatmap(heatmap)


def generate_smoothgradcam_heatmap(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    class_index: int,
    target_layer_name: str | None = None,
    noise_samples: int = 8,
    noise_sigma: float = 0.1,
) -> np.ndarray:
    return generate_smoothgrad_heatmap(
        model=model,
        input_tensor=input_tensor,
        class_index=class_index,
        target_layer_name=target_layer_name,
        noise_samples=noise_samples,
        noise_sigma=noise_sigma,
    )


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
    model_device = infer_model_device(base_model)

    for sample in samples:
        tensor = sample["input_tensor"].detach()
        preview_tensor = tensor.cpu()
        model_tensor = tensor.to(model_device)
        predicted_index = int(sample["predicted_index"])
        image = _tensor_to_uint8_image(preview_tensor, mean=mean, std=std)
        gradcam_heatmap = generate_gradcam_heatmap(
            base_model,
            model_tensor,
            predicted_index,
            target_layer_name=target_layer_name,
        )
        smoothgrad_heatmap = generate_smoothgrad_heatmap(
            base_model,
            model_tensor,
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
                "smoothgrad_heatmap": smoothgrad_heatmap,
            }
        )

    return save_cam_bundle(output_dir=output_dir, overlays=overlays)
