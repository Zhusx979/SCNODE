# SCNODE Rename And Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename unclear files inside `SCNODE`, reorganize source code and historical assets by responsibility, and keep the active training entrypoints runnable.

**Architecture:** Split `SCNODE` into clear areas for training, models, diagnostics, and archives. Move ambiguous top-level scripts and model files into descriptive package paths, update imports to the new locations, and archive legacy datasets/results under dedicated folders.

**Tech Stack:** Python, pathlib, pytest

---

### Task 1: Lock the new SCNODE layout with tests

**Files:**
- Create: `tests/test_scnode_structure.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path


def test_scnode_uses_descriptive_training_and_model_paths():
    scnode_root = Path("SCNODE")
    expected_paths = [
        scnode_root / "training" / "run_bm_experiment.py",
        scnode_root / "training" / "classification_trainer.py",
        scnode_root / "training" / "experiment_config.py",
        scnode_root / "training" / "run_cifar10_experiment.py",
        scnode_root / "diagnostics" / "inspect_ode_attention.py",
        scnode_root / "models" / "cnn" / "resnet18_family.py",
        scnode_root / "models" / "baselines" / "efficientnet_baseline.py",
        scnode_root / "models" / "ode" / "odenet_variants.py",
    ]

    missing = [str(path) for path in expected_paths if not path.exists()]
    assert not missing, missing
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_scnode_structure.py::test_scnode_uses_descriptive_training_and_model_paths -v`
Expected: FAIL because the new directories and names do not exist yet

- [ ] **Step 3: Expand the test**

```python
def test_scnode_archives_legacy_results_and_datasets():
    ...


def test_old_ambiguous_top_level_filenames_are_removed():
    ...
```

- [ ] **Step 4: Run focused tests**

Run: `pytest tests/test_scnode_structure.py -v`
Expected: FAIL with missing paths

### Task 2: Rename source files into clear package paths

**Files:**
- Create: `SCNODE/training/__init__.py`
- Create: `SCNODE/models/__init__.py`
- Create: `SCNODE/models/cnn/__init__.py`
- Create: `SCNODE/models/baselines/__init__.py`
- Create: `SCNODE/models/ode/__init__.py`
- Create: `SCNODE/models/ode/scnode/__init__.py`
- Create: `SCNODE/models/ode/scnode/anode_v2/__init__.py`
- Create: `SCNODE/diagnostics/__init__.py`
- Move: `SCNODE/Train_Test_Val.py` -> `SCNODE/training/run_bm_experiment.py`
- Move: `SCNODE/Function_Train_Test_Val.py` -> `SCNODE/training/classification_trainer.py`
- Move: `SCNODE/config.py` -> `SCNODE/training/experiment_config.py`
- Move: `SCNODE/cifar10_train.py` -> `SCNODE/training/run_cifar10_experiment.py`
- Move: `SCNODE/check.py` -> `SCNODE/diagnostics/inspect_ode_attention.py`
- Move: `SCNODE/model/ResNet18_Series.py` -> `SCNODE/models/cnn/resnet18_family.py`
- Move: `SCNODE/model/ResNet32_Series.py` -> `SCNODE/models/cnn/resnet32_family.py`
- Move: `SCNODE/model/ResNet50_Series.py` -> `SCNODE/models/cnn/resnet50_family.py`
- Move: `SCNODE/model/FcaNet.py` -> `SCNODE/models/cnn/fcanet_backbone.py`
- Move: `SCNODE/model/deit.py` -> `SCNODE/models/cnn/deit_backbone.py`
- Move: `SCNODE/model/pvt.py` -> `SCNODE/models/cnn/pvt_backbone.py`
- Move: `SCNODE/model/tnt.py` -> `SCNODE/models/cnn/tnt_backbone.py`
- Move: `SCNODE/compare_experience/Densenet.py` -> `SCNODE/models/baselines/densenet_baseline.py`
- Move: `SCNODE/compare_experience/EffientNet.py` -> `SCNODE/models/baselines/efficientnet_baseline.py`
- Move: `SCNODE/compare_experience/Mobilenet.py` -> `SCNODE/models/baselines/mobilenet_baseline.py`
- Move: `SCNODE/compare_experience/VGG16.py` -> `SCNODE/models/baselines/vgg16_baseline.py`
- Move: `SCNODE/compare_experience/VIT.py` -> `SCNODE/models/baselines/vision_transformer_baseline.py`
- Move: `SCNODE/ODEmodel/ODE_Series.py` -> `SCNODE/models/ode/odenet_variants.py`
- Move: `SCNODE/ODEmodel/origin_odenet.py` -> `SCNODE/models/ode/odenet_reference.py`
- Move: `SCNODE/ODEmodel/ODE-origin-train-val-test.py` -> `SCNODE/models/ode/legacy_ode_train_eval.py`
- Move: `SCNODE/ODEmodel/paper_model/SCNODE.py` -> `SCNODE/models/ode/scnode/scnode_resnet.py`
- Move: `SCNODE/ODEmodel/paper_model/anode.py` -> `SCNODE/models/ode/scnode/anode_model.py`
- Move: `SCNODE/ODEmodel/paper_model/anode_series.py` -> `SCNODE/models/ode/scnode/anode_variants.py`
- Move: `SCNODE/ODEmodel/paper_model/AnodeV2/AnodeV2_ResNet.py` -> `SCNODE/models/ode/scnode/anode_v2/anode_v2_resnet.py`
- Move: `SCNODE/ODEmodel/paper_model/AnodeV2/AnodeV2_Sqnxt.py` -> `SCNODE/models/ode/scnode/anode_v2/anode_v2_squeezenext.py`

- [ ] **Step 1: Perform the moves**

Run: `Move-Item ...`

- [ ] **Step 2: Update imports**

```python
from SCNODE.training.experiment_config import args
from SCNODE.models.cnn.resnet18_family import Get_ResNet18
```

- [ ] **Step 3: Run structure tests**

Run: `pytest tests/test_scnode_structure.py -v`
Expected: PASS

### Task 3: Archive legacy data and result folders inside SCNODE

**Files:**
- Create: `SCNODE/archive/README.md`
- Move: `SCNODE/all_dataset` -> `SCNODE/archive/datasets/all_dataset`
- Move: `SCNODE/cifar10_data` -> `SCNODE/archive/datasets/cifar10_data`
- Move: `SCNODE/BM_experiment_result` -> `SCNODE/archive/results/bm_experiment_result`
- Move: `SCNODE/cifar10_experiment_result` -> `SCNODE/archive/results/cifar10_experiment_result`

- [ ] **Step 1: Move legacy assets into archive folders**

Run: `Move-Item ...`

- [ ] **Step 2: Document the archive purpose**

```markdown
`archive/datasets/` stores frozen legacy datasets kept only for reference.
`archive/results/` stores old experiment outputs that are no longer the active artifact path.
```

- [ ] **Step 3: Run structure tests**

Run: `pytest tests/test_scnode_structure.py -v`
Expected: PASS

### Task 4: Verify imports and entrypoints

**Files:**
- Modify: `README.md`
- Modify: `SCNODE/__init__.py`

- [ ] **Step 1: Update README references**

```markdown
- `SCNODE/training/run_bm_experiment.py`
- `SCNODE/training/experiment_config.py`
- `SCNODE/models/...`
- `SCNODE/archive/...`
```

- [ ] **Step 2: Run verification**

Run: `pytest tests/test_scnode_structure.py tests/test_data_pipeline.py tests/test_evaluation_metrics.py tests/test_visual_outputs.py -v`
Expected: PASS

- [ ] **Step 3: Run compilation checks**

Run: `python -m py_compile SCNODE\\training\\run_bm_experiment.py SCNODE\\training\\classification_trainer.py SCNODE\\training\\experiment_config.py`
Expected: exit 0
