# SCNODE

SCNODE is a continuous-depth neural network framework for bone-marrow cytomorphology image classification. The proposed model combines convolutional feature extraction with neural ordinary differential equation blocks and exposes the temporal components used in the method, including time-conditioned convolution, time-windowed batch normalization, state augmentation, and SAM. The repository also contains the NODE, ANODE, ANODEV2, ResNet, and other comparison implementations used for the paper experiments.

## Project structure

```text
`-- SCNODE/
    |-- models/
    |   |-- ode/
    |   |   |-- scnode/              SCNODE, ANODE, ANODEV2, TConv, TW-BN, SAM
    |   |   |-- odenet_variants.py   NODE comparison models
    |   |   `-- odenet_reference.py  ODE reference implementation
    |   |-- cnn/                     ResNet and CNN comparison backbones
    |   `-- baselines/               additional CNN and transformer baselines
    |-- training/                    reusable training code and experiment entry points
    |-- experiments/                 condition grids and experiment lifecycle
    |-- diagnostics/                 trajectories, solver checks, metrics, and statistics
    |-- visualization/               multi-step UMAP and figures
    |-- blood_experiment/            BM data loading, evaluation, CAM, and plotting
    `-- tests/                       model, configuration, data, diagnostic, and output tests
|-- requirements-dev.txt            development and test dependencies
`-- README.md                       project overview and source map
```

```

## Model organization

The reviewer-facing model names are registered in `SCNODE/training/experiment_config.py`:

- `SCNODE_ResNet18`, `SCNODE_ResNet34`, and `SCNODE_ResNet50` are the three SCNODE backbone variants.
- `ANODEV2_ResNet18` is the ANODEV2 comparison implementation.
- `ANODE` and `NODE` are the augmented neural ODE and neural ODE comparison models.
- `ResNet18`, `ResNet32`, and `ResNet50` are conventional convolutional comparison models.

The shared SCNODE architecture settings are defined in `SCNODE/models/ode/scnode/config.py`. The configuration contains the solver, temporal grid, state augmentation, downsampling, TConv, TW-BN, SAM, and BN switches used by the ablation tables.

## Experiment organization

`SCNODE/training/` contains the BM cytomorphology and CIFAR-10 training programs and the reusable classification trainer. `SCNODE/experiments/` defines the paper main-table conditions, A-E architecture ablations, and independent BN/SAM/TConv/TW-BN component ablations. Each condition is represented as explicit configuration data so that the selected model and settings can be audited independently of generated results.

## Diagnostics and visualization

`SCNODE/diagnostics/trajectory_experiment.py` provides a common trajectory interface for SCNODE, ANODE, and NODE. It returns logits together with per-ODE-block solver states, allowing the same downstream analysis to be applied to all continuous-depth models.

`SCNODE/diagnostics/statistics.py` contains the two-sided paired t-test and Benjamini-Hochberg FDR procedure used for the paper comparisons. The remaining diagnostic modules provide solver metrics, trajectory summaries, and ODE inspection utilities.

`SCNODE/visualization/` contains the multi-step UMAP workflow and figure entry points. `SCNODE/blood_experiment/visualization.py` contains BM classification figures such as confusion matrices and metric plots.

## Data and generated artifacts

The BM dataset is not included in this repository. Its loader accepts an external class-folder dataset and creates local split metadata under the ignored `SCNODE/artifacts/` directory. CIFAR-10 is accessed through torchvision and is likewise kept under the local artifacts directory. No dataset, checkpoint, log, private path, user-identifying metadata, or generated result is part of the review source tree.

The source code defines the procedures needed to regenerate the paper tables and figures; numerical results require the corresponding dataset, selected seeds, and trained checkpoints.
```
