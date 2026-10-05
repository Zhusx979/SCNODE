from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from SCNODE.blood_experiment.data import get_default_raw_dataset_root


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


@dataclass(frozen=True)
class ExperimentRuntimeConfig:

    output_root: Path | str = "artifacts/experiments"
    learning_rate: float = 1e-3
    optimizer: str = "adam"
    scheduler: str = "none"
    weight_decay: float = 1e-4
    momentum: float = 0.9
    min_learning_rate: float = 0.0
    balance_to_3000: bool = True
    paper_augmentation: bool = True
    max_samples_per_class: int = 6000
    use_tqdm: bool = True
    train_log_interval: int = 20
    is_ode: bool = False
    full_report: bool = True
    save_report_pth: bool = True
    generate_visualizations: bool = True
    generate_cam: bool = True
    cam_samples_per_class: int = 1
    cam_noise_samples: int = 8
    cam_noise_sigma: float = 0.1
    cam_target_layer: str = ""
    collect_ode_diagnostics: bool = False
    evaluate_test_each_epoch: bool = False

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["output_root"] = str(payload["output_root"])
        return payload


def runtime_config_from_args(parsed_args: argparse.Namespace) -> ExperimentRuntimeConfig:
    return ExperimentRuntimeConfig(
        output_root=parsed_args.folder_name,
        learning_rate=parsed_args.lr,
        optimizer=parsed_args.optimizer,
        scheduler=parsed_args.scheduler,
        weight_decay=parsed_args.weight_decay,
        momentum=parsed_args.momentum,
        min_learning_rate=parsed_args.min_learning_rate,
        balance_to_3000=parsed_args.balance_to_3000,
        paper_augmentation=parsed_args.paper_augmentation,
        max_samples_per_class=parsed_args.max_samples_per_class,
        use_tqdm=parsed_args.use_tqdm,
        train_log_interval=parsed_args.train_log_interval,
        is_ode=parsed_args.is_ode,
        full_report=parsed_args.full_report,
        save_report_pth=parsed_args.save_report_pth,
        generate_visualizations=parsed_args.generate_visualizations,
        generate_cam=parsed_args.generate_cam,
        cam_samples_per_class=parsed_args.cam_samples_per_class,
        cam_noise_samples=parsed_args.cam_noise_samples,
        cam_noise_sigma=parsed_args.cam_noise_sigma,
        cam_target_layer=parsed_args.cam_target_layer,
        collect_ode_diagnostics=parsed_args.collect_ode_diagnostics or parsed_args.is_ode,
        evaluate_test_each_epoch=False,
    )


AVAILABLE_MODELS = {
    "ResNet18": ModelSpec("ResNet18", "SCNODE.models.cnn.resnet18_family", "Get_ResNet18"),
    "ResNet32": ModelSpec("ResNet32", "SCNODE.models.cnn.resnet32_family", "Get_ResNet32"),
    "ResNet50": ModelSpec("ResNet50", "SCNODE.models.cnn.resnet50_family", "Get_ResNet50"),
    "NODE": ModelSpec("NODE", "SCNODE.models.ode.odenet_variants", "Get_odenet18"),
    "ANODE": ModelSpec("ANODE", "SCNODE.models.ode.scnode.anode_variants", "Get_aodenet18"),
    "FCANet18": ModelSpec("FCANet18", "SCNODE.models.cnn.fcanet_backbone", "fcanet18"),
    "FCANet34": ModelSpec("FCANet34", "SCNODE.models.cnn.fcanet_backbone", "fcanet34"),
    "FCANet50": ModelSpec("FCANet50", "SCNODE.models.cnn.fcanet_backbone", "fcanet50"),
    "SwinT": ModelSpec("SwinT", "SCNODE.models.baselines.timm_vision_baselines", "Get_SwinT"),
    "SwinS": ModelSpec("SwinS", "SCNODE.models.baselines.timm_vision_baselines", "Get_SwinS"),
    "SwinB": ModelSpec("SwinB", "SCNODE.models.baselines.timm_vision_baselines", "Get_SwinB"),
    "ConvNeXtT": ModelSpec("ConvNeXtT", "SCNODE.models.baselines.timm_vision_baselines", "Get_ConvNeXtT"),
    "ConvNeXtS": ModelSpec("ConvNeXtS", "SCNODE.models.baselines.timm_vision_baselines", "Get_ConvNeXtS"),
    "ConvNeXtB": ModelSpec("ConvNeXtB", "SCNODE.models.baselines.timm_vision_baselines", "Get_ConvNeXtB"),
    "DeiTTiny": ModelSpec("DeiTTiny", "SCNODE.models.baselines.timm_vision_baselines", "Get_DeiTTiny"),
    "DeiTSmall": ModelSpec("DeiTSmall", "SCNODE.models.baselines.timm_vision_baselines", "Get_DeiTSmall"),
    "DeiTBase": ModelSpec("DeiTBase", "SCNODE.models.baselines.timm_vision_baselines", "Get_DeiTBase"),
    "TNTS": ModelSpec("TNTS", "SCNODE.models.cnn.tnt_backbone", "tnt_s_patch16_224"),
    "TNTB": ModelSpec("TNTB", "SCNODE.models.cnn.tnt_backbone", "tnt_b_patch16_224"),
    "SCNODE_ResNet18": ModelSpec(
        "SCNODE_ResNet18",
        "SCNODE.models.ode.scnode.scnode_resnet",
        "Get_time_AnodeV2_ResNet18",
    ),
    "SCNODE_ResNet34": ModelSpec(
        "SCNODE_ResNet34", "SCNODE.models.ode.scnode.scnode_resnet", "Get_time_AnodeV2_ResNet34"
    ),
    "SCNODE_ResNet50": ModelSpec(
        "SCNODE_ResNet50", "SCNODE.models.ode.scnode.scnode_resnet", "Get_time_AnodeV2_ResNet50"
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
    parser.add_argument("--optimizer", choices=["adam", "adamw", "sgd", "rmsprop", "adagrad"], default="adam")
    parser.add_argument("--scheduler", choices=["none", "cosine"], default="none")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--min_learning_rate", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--test_batch_size", type=int, default=256)
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
    parser.add_argument("--collect_ode_diagnostics", type=_bool_argument, default=False)
    parser.add_argument("--folder_name", type=str, default="artifacts/experiments")
    parser.add_argument("--raw_data_root", type=str, default=str(get_default_raw_dataset_root()))
    parser.add_argument("--prepared_data_root", type=str, default="artifacts/datasets/bm_21class_split")
    parser.add_argument("--balance_to_3000", type=_bool_argument, default=True)
    parser.add_argument("--paper_augmentation", type=_bool_argument, default=True)
    parser.add_argument("--max_samples_per_class", type=int, default=6000)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--model_names", nargs="+", default=["ResNet18"])
    parser.add_argument(
        "--loss",
        choices=["cross_entropy", "ldam_drw", "logit_adjustment", "balanced_softmax", "class_balanced_focal", "seesaw", "eql_v2"],
        default="cross_entropy",
    )
    return parser


def get_selected_models(model_names: list[str]) -> list[tuple[ModelSpec, str]]:
    selected_models = []
    for model_name in model_names:
        if model_name not in AVAILABLE_MODELS:
            supported = ", ".join(sorted(AVAILABLE_MODELS))
            raise ValueError(f"Unknown model '{model_name}'. Supported models: {supported}")
        selected_models.append((AVAILABLE_MODELS[model_name], model_name))
    return selected_models


def parse_experiment_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[tuple[ModelSpec, str]]]:
    parsed_args = build_parser().parse_args(argv)
    return parsed_args, get_selected_models(parsed_args.model_names)
