import argparse
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from blood_experiment.data import get_default_raw_dataset_root


@dataclass(frozen=True)
class ModelSpec:
    name: str
    module_path: str
    factory_name: str

    def load_factory(self):
        try:
            module = importlib.import_module(self.module_path)
        except ImportError as exc:
            raise RuntimeError(
                f"Failed to import model module '{self.module_path}' for '{self.name}'. "
                f"Install the missing dependency first. Original error: {exc}"
            ) from exc
        return getattr(module, self.factory_name)


AVAILABLE_MODELS = {
    "ResNet18": ModelSpec("ResNet18", "SCNODE.models.cnn.resnet18_family", "Get_ResNet18"),
    "ResNet32": ModelSpec("ResNet32", "SCNODE.models.cnn.resnet32_family", "Get_ResNet32"),
    "ResNet50": ModelSpec("ResNet50", "SCNODE.models.cnn.resnet50_family", "Get_ResNet50"),
    "ODENet18": ModelSpec("ODENet18", "SCNODE.models.ode.odenet_variants", "Get_odenet18"),
    "ODENet34": ModelSpec("ODENet34", "SCNODE.models.ode.odenet_variants", "Get_odenet34"),
    "ODENet50": ModelSpec("ODENet50", "SCNODE.models.ode.odenet_variants", "Get_odenet50"),
    "SCNODE_ResNet18": ModelSpec(
        "SCNODE_ResNet18",
        "SCNODE.models.ode.scnode.scnode_resnet",
        "Get_time_AnodeV2_ResNet18",
    ),
    "AnodeV2_ResNet18": ModelSpec(
        "AnodeV2_ResNet18",
        "SCNODE.models.ode.scnode.anode_v2.anode_v2_resnet",
        "Get_AnodeV2_ResNet18",
    ),
    "AnodeV2_ResNet34": ModelSpec(
        "AnodeV2_ResNet34",
        "SCNODE.models.ode.scnode.anode_v2.anode_v2_resnet",
        "Get_AnodeV2_ResNet34",
    ),
    "AnodeV2_ResNet50": ModelSpec(
        "AnodeV2_ResNet50",
        "SCNODE.models.ode.scnode.anode_v2.anode_v2_resnet",
        "Get_AnodeV2_ResNet50",
    ),
}


def _bool_argument(value):
    if isinstance(value, bool):
        return value
    if value in {"True", "true", "1"}:
        return True
    if value in {"False", "false", "0"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_log_interval", type=int, default=20)
    parser.add_argument("--use_tqdm", type=_bool_argument, default=True)
    parser.add_argument("--gpus", type=_bool_argument, default=True)
    parser.add_argument("--is_ode", type=_bool_argument, default=False, choices=[True, False])
    parser.add_argument("--full_report", type=_bool_argument, default=True, choices=[True, False])
    parser.add_argument("--save_report_pth", type=_bool_argument, default=True, choices=[True, False])
    parser.add_argument("--generate_visualizations", type=_bool_argument, default=True, choices=[True, False])
    parser.add_argument("--generate_cam", type=_bool_argument, default=True, choices=[True, False])
    parser.add_argument("--cam_samples_per_class", type=int, default=1)
    parser.add_argument("--cam_noise_samples", type=int, default=8)
    parser.add_argument("--cam_noise_sigma", type=float, default=0.1)
    parser.add_argument("--cam_target_layer", type=str, default="")
    parser.add_argument("--folder_name", type=str, default="artifacts/experiments")
    parser.add_argument("--raw_data_root", type=str, default=str(get_default_raw_dataset_root()))
    parser.add_argument("--prepared_data_root", type=str, default="artifacts/datasets/bm_21class_split")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--model_names", nargs="+", default=["ResNet18"])
    return parser


def get_selected_models(model_names: list[str]) -> list[tuple[ModelSpec, str]]:
    selected_models = []
    for model_name in model_names:
        if model_name not in AVAILABLE_MODELS:
            supported = ", ".join(sorted(AVAILABLE_MODELS))
            raise ValueError(f"Unknown model '{model_name}'. Supported models: {supported}")
        selected_models.append((AVAILABLE_MODELS[model_name], model_name))
    return selected_models


parser = build_parser()
args = parser.parse_args()
models = get_selected_models(args.model_names)
