# Blood Experiment Workspace

当前目录已经按“活跃实验链路”和“历史模型代码”分开理解：

- `BM_cytomorphology_data/`
  当前骨髓细胞原始数据集，21 个类别，实验默认直接从这里读取。
- `blood_experiment/`
  新增的活跃实验工具层，负责数据分割清单、每类评估指标、ROC/PR 曲线、混淆矩阵、Grad-CAM 与 SmoothGrad-CAM。
- `SCNODE/`
  已整理为 `training/`、`models/`、`diagnostics/`、`archive/` 四层结构；活跃训练入口仍在这里，并已经接入 `blood_experiment/`。
- `tests/`
  针对数据拆分、指标计算和图像产物的回归测试。
- `docs/superpowers/plans/`
  本次重构实施计划。
- `artifacts/`
  运行训练后生成的分割清单、数据摘要、评估指标、图像曲线和 CAM 可视化。

## 当前活跃入口

运行：

```bash
python SCNODE/training/run_bm_experiment.py --model_names ResNet18
```

常用参数：

```bash
python SCNODE/training/run_bm_experiment.py ^
  --model_names ResNet18 SCNODE_ResNet18 ^
  --raw_data_root "E:\School Work\Deep Learning\Paper\blood\code\BM_cytomorphology_data" ^
  --prepared_data_root "artifacts/datasets/bm_21class_split" ^
  --folder_name "artifacts/experiments"
```

## 训练后会生成的主要结果

- `artifacts/datasets/bm_21class_split/split_manifest.csv`
- `artifacts/datasets/bm_21class_split/dataset_summary.csv`
- `artifacts/experiments/<model_name>/metrics/`
- `artifacts/experiments/<model_name>/plots/`
- `artifacts/experiments/<model_name>/cam/`
