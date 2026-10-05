# SCNODE

SCNODE is a continuous-depth neural network codebase for cell image recognition. It combines convolutional feature extraction with Neural ODEs to evolve feature representations through a continuous-time state space. The codebase also includes time-conditioned convolution (TConv), temporal-window batch normalization (TW-BN),  and SAM components. These components support the complete SCNODE design and unified comparisons with NODE, ANODE, ANODEV2 and conventional convolutional networks.

## Project components

| Module               | Role                                                                                                     |
| -------------------- | -------------------------------------------------------------------------------------------------------- |
| `models`           | SCNODE, NODE, ANODE, ANODEV2, convolutional models and vision baselines                                  |
| `training`         | Classification trainers, BM and CIFAR training entry points, optimizers and training logs                |
| `experiments`      | Main-table conditions, structural ablations, component ablations and experiment lifecycle configurations |
| `diagnostics`      | ODE solver checks, feature trajectories, runtime metrics and statistical analysis                        |
| `visualization`    | Multi-step UMAP and continuous-depth model trajectory figures                                            |
| `blood_experiment` | Cell image data processing, augmentation, evaluation, confusion matrices and CAM                         |
| `tests`            | Tests for models, data pipelines, configurations, training outputs and diagnostics                       |

## Directory structure

```text
SCNODE/
├── models/
│   ├── ode/
│   │   ├── scnode/
│   │   │   ├── scnode.py
│   │   │   ├── scnode_resnet.py
│   │   │   ├── config.py
│   │   │   ├── anode_model.py
│   │   │   └── anode_variants.py
│   │   ├── odenet_variants.py
│   │   ├── odenet_reference.py
│   │   └── legacy_ode_train_eval.py
│   ├── cnn/
│   └── baselines/
├── training/
│   ├── classification_trainer.py
│   ├── experiment_config.py
│   ├── run_bm_balanced_aug_experiment.py
│   ├── run_bm_experiment_selectable.py
│   ├── run_bm_experiment.py
│   ├── run_cifar10_experiment.py
│   ├── optimizers.py
│   └── progress_reporting.py
├── experiments/
├── diagnostics/
├── visualization/
├── blood_experiment/
│   ├── bm_balanced.py
│   ├── data.py
│   ├── evaluation.py
│   ├── visualization.py
│   └── cam.py
├── tests/
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

## Model implementation

`models/ode/scnode/scnode.py` contains the main SCNODE network definitions. Ordinary residual blocks use standard `BatchNorm2d`. ODE function blocks use time-conditioned convolution and TW-BN. TW-BN maintains separate statistics at 11 temporal grid points and smooths them with a window of length 5. `scnode_resnet.py` provides a compatibility export interface so that existing experiment configurations can continue to use consistent model names.

Model registration is centralized in `training/experiment_config.py`. The codebase currently includes the following comparison models used in the paper:

| Registered name                          | Model role                            |
| ---------------------------------------- | ------------------------------------- |
| `SCNODE_ResNet18`                      | Main SCNODE model                     |
| `SCNODE_ResNet34`                      | SCNODE depth variant                  |
| `SCNODE_ResNet50`                      | SCNODE depth variant                  |
| `NODE`                                 | Neural ODE comparison model           |
| `ANODE`                                | Augmented Neural ODE comparison model |
| `ANODEV2_ResNet18`                     | ANODEV2 comparison model              |
| `ResNet18`, `ResNet32`, `ResNet50` | Conventional convolutional baselines  |

`models/ode/scnode/config.py` provides shared control over the solver, temporal grid, state augmentation, downsampling, TConv, TWBN, SAM and BN switches. This configuration structure corresponds to the experimental factors in the main and ablation tables.

## Training and data modules

`blood_experiment/data.py` handles class-directory scanning, manifest generation, deterministic data splitting and evaluation preprocessing. `blood_experiment/bm_balanced.py` provides the class-balanced training dataset for BM experiments. Training samples are resampled and augmented after the 8:1:1 split. Validation and test samples use deterministic preprocessing only.

`training/run_bm_balanced_aug_experiment.py` is the complete entry point for BM cell image experiments. It provides training logs, validation-accuracy-based model selection, checkpoint recovery, CSV records and final test evaluation. `training/run_bm_experiment_selectable.py` provides selectable no-augmentation and training-set augmentation modes. `training/run_cifar10_experiment.py` handles CIFAR image classification experiments.

## Code quality

`tests/` covers model structure, TWBN state, data splitting, augmentation pipelines, training outputs, statistical configurations and visualization outputs. The test suite verifies code behaviour and experimental configurations.
