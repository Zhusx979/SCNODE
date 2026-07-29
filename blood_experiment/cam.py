from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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


def _activation_alpha(normalized_heatmap: np.ndarray) -> np.ndarray:
    return np.clip(np.power(normalized_heatmap, 0.8), 0.14, 0.82)


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


def save_overlay_preview(
    output_path: Path | str,
    image: np.ndarray,
    heatmap: np.ndarray,
    title: str,
) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    base_image = _prepare_base_image(image)
    normalized_heatmap = _normalize_heatmap(heatmap)

    fig, ax = plt.subplots(figsize=(4.9, 4.9), dpi=220)
    overlay_artist = _draw_cam_panel(ax, base_image, normalized_heatmap, title)
    colorbar = fig.colorbar(overlay_artist, ax=ax, fraction=0.045, pad=0.03, shrink=0.88)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(labelsize=8.5)
    colorbar.set_label("Activation", fontsize=9)
    fig.tight_layout()
    saved_path = finalize_figure(fig, destination)
    plt.close(fig)
    return saved_path


def save_cam_comparison_figure(
    output_path: Path | str,
    image: np.ndarray,
    gradcam_heatmap: np.ndarray,
    smoothgradcam_heatmap: np.ndarray,
    predicted_label: str,
    true_label: str,
) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    base_image = _prepare_base_image(image)
    normalized_gradcam = _normalize_heatmap(gradcam_heatmap)
    normalized_smooth = _normalize_heatmap(smoothgradcam_heatmap)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(10.8, 3.9),
        dpi=220,
        gridspec_kw={"wspace": 0.03},
        constrained_layout=True,
    )
    _draw_cam_panel(axes[0], base_image, None, "Input Image")
    grad_artist = _draw_cam_panel(axes[1], base_image, normalized_gradcam, f"Grad-CAM | Pred: {predicted_label}")
    _draw_cam_panel(axes[2], base_image, normalized_smooth, f"Smooth Grad-CAM | Pred: {predicted_label}")

    axes[0].text(
        0.02,
        0.04,
        f"True: {true_label}",
        transform=axes[0].transAxes,
        fontsize=9.3,
        color="#F8FAFC",
        bbox={"boxstyle": "round,pad=0.25", "fc": "#1E293B", "ec": "#334155", "lw": 0.8, "alpha": 0.92},
    )

    colorbar = fig.colorbar(grad_artist, ax=axes, fraction=0.02, pad=0.01, shrink=0.92)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(labelsize=8.5)
    colorbar.set_label("Activation intensity", fontsize=9)

    fig.suptitle("Class Activation Comparison", x=0.055, y=1.03, ha="left", fontsize=13.5, fontweight="bold")
    saved_path = finalize_figure(fig, destination)
    plt.close(fig)
    return saved_path


@dataclass
class CamArtifact:
    image_path: str
    predicted_label: str
    true_label: str
    gradcam_path: Path
    smoothgradcam_path: Path
    comparison_path: Path | None = None


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
        comparison_path = save_cam_comparison_figure(
            destination / f"sample_{index:03d}_cam_comparison.png",
            image=overlay["image"],
            gradcam_heatmap=overlay["gradcam_heatmap"],
            smoothgradcam_heatmap=overlay["smoothgradcam_heatmap"],
            predicted_label=str(overlay.get("predicted_label", "")),
            true_label=str(overlay.get("true_label", "")),
        )
        artifacts.append(
            CamArtifact(
                image_path=str(overlay.get("image_path", "")),
                predicted_label=str(overlay.get("predicted_label", "")),
                true_label=str(overlay.get("true_label", "")),
                gradcam_path=gradcam_path,
                smoothgradcam_path=smoothgradcam_path,
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
        smoothgradcam_heatmap = generate_smoothgradcam_heatmap(
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
                "smoothgradcam_heatmap": smoothgradcam_heatmap,
                "gradcam_title": f"Grad-CAM | Pred: {class_names[predicted_index]}",
                "smoothgradcam_title": f"Smooth Grad-CAM | Pred: {class_names[predicted_index]}",
            }
        )

    return save_cam_bundle(output_dir=output_dir, overlays=overlays)
