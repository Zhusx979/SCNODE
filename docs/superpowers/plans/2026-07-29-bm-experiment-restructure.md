# BM Experiment Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the active bone-marrow experiment pipeline so it reads directly from `E:\School Work\Deep Learning\Paper\blood\code\BM_cytomorphology_data`, outputs richer per-class metrics and publication-ready figures, and leaves the legacy `SCNODE` code usable.

**Architecture:** Add a focused `blood_experiment` package for dataset splitting, metrics, curves, and CAM visualizations, then wire the existing `SCNODE/Train_Test_Val.py` and `SCNODE/Function_Train_Test_Val.py` into that package. Keep legacy model definitions in place, but move the active experiment flow to clean output directories under `artifacts/`.

**Tech Stack:** Python, PyTorch, scikit-learn, matplotlib, pytest

---

### Task 1: Add dataset path and split utilities

**Files:**
- Create: `blood_experiment/__init__.py`
- Create: `blood_experiment/data.py`
- Test: `tests/test_data_pipeline.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from blood_experiment.data import create_split_manifest, discover_class_names


def test_create_split_manifest_generates_three_splits(tmp_path: Path):
    raw_root = tmp_path / "raw"
    for class_name in ("ABE", "BLA"):
        for index in range(8):
            image_dir = raw_root / class_name / "0001-1000"
            image_dir.mkdir(parents=True, exist_ok=True)
            (image_dir / f"{class_name}_{index:04d}.jpg").write_bytes(b"fake")

    manifest = create_split_manifest(
        raw_root=raw_root,
        output_dir=tmp_path / "prepared",
        train_ratio=0.75,
        val_ratio=0.125,
        test_ratio=0.125,
        seed=7,
    )

    assert manifest.exists()
    assert discover_class_names(raw_root) == ["ABE", "BLA"]
    content = manifest.read_text(encoding="utf-8")
    assert "train" in content
    assert "val" in content
    assert "test" in content
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_data_pipeline.py::test_create_split_manifest_generates_three_splits -v`
Expected: FAIL with `ModuleNotFoundError` or missing function errors for `blood_experiment.data`

- [ ] **Step 3: Write minimal implementation**

```python
from pathlib import Path


def discover_class_names(raw_root: Path) -> list[str]:
    return sorted(path.name for path in Path(raw_root).iterdir() if path.is_dir())


def create_split_manifest(raw_root: Path, output_dir: Path, train_ratio: float, val_ratio: float, test_ratio: float, seed: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "split_manifest.csv"
    manifest_path.write_text("split,class_name,class_index,image_path\n", encoding="utf-8")
    return manifest_path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_data_pipeline.py::test_create_split_manifest_generates_three_splits -v`
Expected: PASS

- [ ] **Step 5: Expand implementation**

```python
def allocate_split_counts(sample_count: int, train_ratio: float, val_ratio: float, test_ratio: float) -> dict[str, int]:
    ...


def discover_image_paths(class_dir: Path) -> list[Path]:
    ...


def create_split_manifest(...):
    ...
```

- [ ] **Step 6: Run focused tests**

Run: `pytest tests/test_data_pipeline.py -v`
Expected: PASS with coverage for class discovery, deterministic splitting, and manifest layout

### Task 2: Add evaluation metrics and report exports

**Files:**
- Create: `blood_experiment/evaluation.py`
- Test: `tests/test_evaluation_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np

from blood_experiment.evaluation import build_evaluation_bundle


def test_build_evaluation_bundle_contains_per_class_metrics():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_prob = np.array(
        [
            [0.90, 0.10],
            [0.20, 0.80],
            [0.30, 0.70],
            [0.10, 0.90],
        ]
    )

    bundle = build_evaluation_bundle(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        class_names=["ABE", "BLA"],
    )

    assert "summary" in bundle
    assert "per_class" in bundle
    assert "roc" in bundle
    assert abs(bundle["summary"]["accuracy"] - 0.75) < 1e-9
    assert bundle["per_class"][0]["class_name"] == "ABE"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_evaluation_metrics.py::test_build_evaluation_bundle_contains_per_class_metrics -v`
Expected: FAIL with `ModuleNotFoundError` or missing symbol errors

- [ ] **Step 3: Write minimal implementation**

```python
from sklearn.metrics import accuracy_score


def build_evaluation_bundle(y_true, y_pred, y_prob, class_names):
    return {
        "summary": {"accuracy": float(accuracy_score(y_true, y_pred))},
        "per_class": [{"class_name": class_name} for class_name in class_names],
        "roc": {},
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_evaluation_metrics.py::test_build_evaluation_bundle_contains_per_class_metrics -v`
Expected: PASS

- [ ] **Step 5: Expand implementation**

```python
def compute_summary_metrics(...):
    ...


def compute_per_class_metrics(...):
    ...


def compute_curve_metrics(...):
    ...
```

- [ ] **Step 6: Run focused tests**

Run: `pytest tests/test_evaluation_metrics.py -v`
Expected: PASS with summary, per-class, and ROC/AUC assertions

### Task 3: Add plotting and CAM utilities

**Files:**
- Create: `blood_experiment/visualization.py`
- Create: `blood_experiment/cam.py`
- Test: `tests/test_visual_outputs.py`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from pathlib import Path

from blood_experiment.visualization import save_confusion_matrix_plot, save_roc_curve_plot


def test_metric_plots_are_written(tmp_path: Path):
    matrix_path = save_confusion_matrix_plot(
        output_path=tmp_path / "cm.png",
        confusion=np.array([[3, 1], [0, 4]]),
        class_names=["ABE", "BLA"],
        normalize=True,
        title="Demo",
    )
    roc_path = save_roc_curve_plot(
        output_path=tmp_path / "roc.png",
        roc_payload={"micro_auc": 0.9, "macro_auc": 0.9, "curves": []},
        class_names=["ABE", "BLA"],
        title="Demo ROC",
    )

    assert matrix_path.exists()
    assert roc_path.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_visual_outputs.py::test_metric_plots_are_written -v`
Expected: FAIL because `blood_experiment.visualization` does not exist

- [ ] **Step 3: Write minimal implementation**

```python
import matplotlib.pyplot as plt


def save_confusion_matrix_plot(output_path, confusion, class_names, normalize, title):
    plt.figure()
    plt.imshow(confusion)
    plt.savefig(output_path)
    plt.close()
    return output_path


def save_roc_curve_plot(output_path, roc_payload, class_names, title):
    plt.figure()
    plt.plot([0, 1], [0, 1])
    plt.savefig(output_path)
    plt.close()
    return output_path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_visual_outputs.py::test_metric_plots_are_written -v`
Expected: PASS

- [ ] **Step 5: Expand implementation**

```python
def save_precision_recall_plot(...):
    ...


def render_gradcam_overlays(...):
    ...


def render_smooth_gradcam_overlays(...):
    ...
```

- [ ] **Step 6: Run focused tests**

Run: `pytest tests/test_visual_outputs.py -v`
Expected: PASS with figure-generation coverage

### Task 4: Integrate active experiment flow with legacy training entrypoints

**Files:**
- Modify: `SCNODE/config.py`
- Modify: `SCNODE/Train_Test_Val.py`
- Modify: `SCNODE/Function_Train_Test_Val.py`
- Test: `tests/test_data_pipeline.py`
- Test: `tests/test_evaluation_metrics.py`
- Test: `tests/test_visual_outputs.py`

- [ ] **Step 1: Write the failing integration test**

```python
from pathlib import Path

from blood_experiment.data import get_default_raw_dataset_root


def test_default_raw_dataset_root_points_to_workspace_dataset():
    dataset_root = get_default_raw_dataset_root()
    assert dataset_root.name == "BM_cytomorphology_data"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_data_pipeline.py::test_default_raw_dataset_root_points_to_workspace_dataset -v`
Expected: FAIL until config helpers are added

- [ ] **Step 3: Write minimal implementation**

```python
from pathlib import Path


def get_default_raw_dataset_root() -> Path:
    return Path(r"E:\School Work\Deep Learning\Paper\blood\code\BM_cytomorphology_data")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_data_pipeline.py::test_default_raw_dataset_root_points_to_workspace_dataset -v`
Expected: PASS

- [ ] **Step 5: Wire training and evaluation**

```python
# SCNODE/config.py
parser.add_argument("--raw_data_root", type=str, default=str(get_default_raw_dataset_root()))
parser.add_argument("--prepared_data_root", type=str, default="artifacts/datasets/bm_split")
parser.add_argument("--experiment_root", type=str, default="artifacts/experiments")
parser.add_argument("--gpus", type=eval, default=False, choices=[True, False])
```

```python
# SCNODE/Train_Test_Val.py
from blood_experiment.data import prepare_experiment_splits

prepared = prepare_experiment_splits(...)
```

```python
# SCNODE/Function_Train_Test_Val.py
from blood_experiment.evaluation import build_evaluation_bundle
from blood_experiment.visualization import save_publication_plots
from blood_experiment.cam import save_cam_bundle
```

- [ ] **Step 6: Run focused tests**

Run: `pytest tests/test_data_pipeline.py tests/test_evaluation_metrics.py tests/test_visual_outputs.py -v`
Expected: PASS

### Task 5: Add usage documentation

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write the doc update**

```markdown
# Blood Experiment Workspace

## Active directories
- `BM_cytomorphology_data/`: raw 21-class dataset
- `blood_experiment/`: active data, metric, and visualization utilities
- `SCNODE/`: legacy model and training code
- `artifacts/`: generated splits, reports, plots, and CAM outputs
- `tests/`: regression tests for the active pipeline
```

- [ ] **Step 2: Verify docs match the final tree**

Run: `Get-ChildItem`
Expected: root contains `BM_cytomorphology_data`, `blood_experiment`, `SCNODE`, `artifacts`, `tests`, `docs`
