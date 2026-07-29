# SCNODE Structure

`SCNODE/` 现在按职责拆成了 4 个主要区域：

- `training/`
  活跃训练入口和训练流程实现。
- `models/`
  按模型性质拆分的 CNN、baseline、ODE/SCNODE 系列模型定义。
- `diagnostics/`
  结构检查和局部实验脚本。
- `archive/`
  历史数据集和历史实验结果，只做参考，不再作为活跃训练默认输出路径。

## Active Entrypoints

- `training/run_bm_experiment.py`
- `training/run_cifar10_experiment.py`
- `training/experiment_config.py`
- `training/classification_trainer.py`

## Model Layout

- `models/cnn/`: ResNet/FcaNet/DeiT/PVT/TNT
- `models/baselines/`: DenseNet/EfficientNet/MobileNet/VGG/ViT
- `models/ode/`: ODENet 变体与 SCNODE/ANODE 系列
