# SCNODE Review Experiment Platform Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `subagent-driven-development` (recommended) or `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver reproducible SCNODE ablations and diagnostics that directly answer reviewer comments 3, 4, 5, 10, and 11, while explicitly limiting claims to evidence available from static labelled cell images.

**Architecture:** Keep ordinary BM training intact. Introduce an explicit SCNODE configuration object, a review-experiment runner, model-independent test-time corruption datasets, ODE/trajectory diagnostics, and seed-level result aggregation. Every review condition writes a resolved configuration, predictions, diagnostics, figures, and aggregate uncertainty under `artifacts/review_experiments/`.

**Tech Stack:** Python 3, PyTorch, torchdiffeq, torchvision, NumPy, scikit-learn, Matplotlib, pytest.

---

## Final File Structure

```text
SCNODE/
  experiments/
    __init__.py
    configs.py                 # condition grids and CLI-to-condition resolution
    runner.py                  # seed runner, dry-run, artifact lifecycle
    summarize.py               # aggregate metrics, bootstrap CI, Markdown tables
    run_review_experiment.py   # user-facing CLI
  diagnostics/
    ode_metrics.py             # hooks, NFE/state/gradient/endpoint metrics
    trajectory.py              # sampled-state export and PCA projection
  models/ode/scnode/
    config.py                  # ScnodeConfig validation
    scnode_resnet.py           # configurable solver, augmentation, time and stem
blood_experiment/
  robustness.py                # deterministic test-only corruptions
tests/
  test_scnode_config.py
  test_ode_diagnostics.py
  test_trajectory.py
  test_robustness.py
  test_review_configs.py
  test_review_summary.py
docs/
  review_experiments.md
```

Do not split `scnode_resnet.py` in this change. It is the active registered factory, and a focused parameterization is lower risk than moving its existing model definitions. New support code belongs in separate modules listed above.

## Common Protocol and Result Contract

Every condition must use the same existing manifest and report exactly these columns in `metrics/summary.json`:

```json
{
  "accuracy": 0.0,
  "macro_f1": 0.0,
  "balanced_accuracy": 0.0,
  "mcc": 0.0,
  "parameter_count": 0,
  "peak_memory_bytes": 0,
  "wall_time_seconds": 0.0,
  "mean_nfe_forward": 0.0,
  "mean_nfe_backward": 0.0,
  "max_state_l2": 0.0,
  "max_gradient_l2": 0.0,
  "nonfinite_batch_count": 0,
  "endpoint_relative_error": 0.0
}
```

The runner uses seeds `42 123 2026`, selects a checkpoint by validation macro-F1, writes test predictions once per selected checkpoint, and aggregates only complete three-seed groups. The aggregate report includes mean, standard deviation, and a paired bootstrap 95% CI for macro-F1 differences. No report may call computational ODE time biological time, or call synthetic colour corruptions cross-site domain adaptation.

## Task 1: Add Validated SCNODE Configuration

**Files:**
- Create: `SCNODE/models/ode/scnode/config.py`
- Modify: `SCNODE/models/ode/scnode/scnode_resnet.py:379-552`
- Create: `tests/test_scnode_config.py`

- [ ] **Step 1: Write failing configuration tests**

```python
import pytest
from SCNODE.models.ode.scnode.config import ScnodeConfig

def test_config_accepts_adaptive_and_fixed_solvers() -> None:
    assert ScnodeConfig(solver="dopri5").solver == "dopri5"
    assert ScnodeConfig(solver="rk4", ode_steps=4).ode_steps == 4

@pytest.mark.parametrize("kwargs", [
    {"solver": "bad"}, {"time_mode": "bad"},
    {"downsampling": "bad"}, {"augment_dim": -1}, {"ode_steps": 0},
])
def test_config_rejects_invalid_values(kwargs) -> None:
    with pytest.raises(ValueError):
        ScnodeConfig(**kwargs)
```

- [ ] **Step 2: Run the test and verify failure**

Run: `pytest tests/test_scnode_config.py -v`

Expected: FAIL because `config.py` does not exist.

- [ ] **Step 3: Implement immutable validated configuration**

```python
@dataclass(frozen=True)
class ScnodeConfig:
    solver: str = "rk4"
    ode_steps: int = 4
    rtol: float = 1e-3
    atol: float = 1e-3
    time_mode: str = "concat"
    augment_dim: int = 1
    downsampling: str = "maxpool"
    ode_entry_size: int = 56

    def __post_init__(self) -> None:
        if self.solver not in {"euler", "rk4", "dopri5"}:
            raise ValueError("solver must be euler, rk4, or dopri5")
        if self.time_mode not in {"none", "concat", "fourier_film"}:
            raise ValueError("time_mode must be none, concat, or fourier_film")
        if self.downsampling not in {"maxpool", "avgpool", "stride_conv"}:
            raise ValueError("downsampling must be maxpool, avgpool, or stride_conv")
        if self.augment_dim < 0 or self.ode_steps < 1:
            raise ValueError("augment_dim must be non-negative and ode_steps positive")
```

- [ ] **Step 4: Parameterize the active SCNODE factory**

Make `Get_time_AnodeV2_ResNet18(num_classes, config: ScnodeConfig | None = None)` accept the configuration. `ODEBlock` must use `torch.linspace(0, 1, ode_steps + 1)` for fixed solvers and `torch.tensor([0, 1])` only for `dopri5`; pass `step_size=1 / ode_steps` for Euler/RK4. Keep the legacy default factory behavior equivalent to `ScnodeConfig()`.

Implement the three time modes inside the ODE function: omit time input for `none`; retain `Conv2dTime` for `concat`; add an MLP from Fourier features `[sin(2*pi*t), cos(2*pi*t)]` to per-channel affine scale/shift for `fourier_film`. Append exactly `augment_dim` zero channels once inside each ODE block; expose `zero_auxiliary=False` in `forward` so the intervention can set them to zero after integration.

- [ ] **Step 5: Verify tensor shapes and solver time grids**

Add tests for a 2x3x224x224 input under all three time modes and all downsamplers. Assert logits have shape `(2, 21)`, each ODE block receives `ode_steps + 1` evaluation times for fixed solvers, and `zero_auxiliary=True` changes only the auxiliary channels before the classifier.

Run: `pytest tests/test_scnode_config.py tests/test_ode_runtime.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add SCNODE/models/ode/scnode/config.py SCNODE/models/ode/scnode/scnode_resnet.py tests/test_scnode_config.py
git commit -m "feat: configure SCNODE ODE experiments"
```

## Task 2: Make Training Outputs Experiment-Ready

**Files:**
- Modify: `SCNODE/training/classification_trainer.py:252-519`
- Modify: `SCNODE/training/ode_runtime.py:1-17`
- Modify: `SCNODE/training/progress_reporting.py:54-65`
- Modify: `SCNODE/training/experiment_config.py:67-99`
- Modify: `SCNODE/training/run_bm_experiment.py:38-90`
- Modify: `tests/test_training_progress_reporting.py`

- [ ] **Step 1: Add failing CSV and explicit-configuration tests**

```python
def test_epoch_metrics_include_ode_diagnostics(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    append_epoch_metrics_row(path, {"epoch": 1, "mean_nfe_forward": 3.0,
                                    "mean_nfe_backward": 4.0,
                                    "max_state_l2": 2.0, "max_gradient_l2": 1.0,
                                    "nonfinite_batch_count": 0})
    assert "mean_nfe_forward" in path.read_text(encoding="utf-8")
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest tests/test_training_progress_reporting.py -v`

Expected: FAIL until training creates the additional fields.

- [ ] **Step 3: Remove global-argument dependency from the reusable trainer**

Add an `ExperimentRuntimeConfig` dataclass passed explicitly to `train_val_test_model`. It contains output path, reporting/CAM switches, and `collect_ode_diagnostics`. Keep `run_bm_experiment.py` as an adapter from current argparse values to that dataclass. Do not import parsed CLI arguments inside the trainer.

- [ ] **Step 4: Persist diagnostics and selected checkpoints**

Write NFE forward/backward, peak CUDA memory, epoch duration, maximum state norm, maximum gradient norm, and non-finite batch count to every epoch CSV row. Save `resolved_config.json`, `best_checkpoint.pt`, and final `test_predictions.npz` in each model/seed directory. Ensure test predictions come from the validation-selected checkpoint, not the final epoch weights.

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_training_progress_reporting.py tests/test_ode_runtime.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add SCNODE/training tests/test_training_progress_reporting.py
git commit -m "feat: persist training diagnostics for review experiments"
```

## Task 3: Add ODE Stability Diagnostics

**Files:**
- Create: `SCNODE/diagnostics/ode_metrics.py`
- Create: `tests/test_ode_diagnostics.py`
- Modify: `SCNODE/training/classification_trainer.py`

- [ ] **Step 1: Write failing diagnostic tests**

```python
def test_endpoint_relative_error_is_zero_for_identical_states() -> None:
    value = endpoint_relative_error(torch.ones(2, 3), torch.ones(2, 3))
    assert value == 0.0

def test_metrics_collector_reports_nonfinite_and_norms() -> None:
    collector = OdeMetricCollector()
    collector.record_state(torch.tensor([1.0, float("nan")]))
    collector.record_gradient(torch.tensor([3.0, 4.0]))
    assert collector.snapshot()["nonfinite_batch_count"] == 1
    assert collector.snapshot()["max_gradient_l2"] == 5.0
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest tests/test_ode_diagnostics.py -v`

Expected: FAIL because the diagnostic module is absent.

- [ ] **Step 3: Implement metric collection**

Implement `OdeMetricCollector` with forward hooks for every active `ODEBlock`, a `record_gradient` method after `loss.backward()`, and `endpoint_relative_error(reference, candidate, eps=1e-12)`. A reference endpoint uses the same trained weights evaluated with `dopri5`, `rtol=atol=1e-5`; it is an integration-consistency diagnostic, not ground truth. Never silently discard non-finite values; record the batch and terminate that seed cleanly with status `failed_nonfinite`.

- [ ] **Step 4: Integrate and test**

Run: `pytest tests/test_ode_diagnostics.py tests/test_scnode_config.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add SCNODE/diagnostics/ode_metrics.py SCNODE/training/classification_trainer.py tests/test_ode_diagnostics.py
git commit -m "feat: add ODE stability diagnostics"
```

## Task 4: Add Deterministic Test-Time Corruptions

**Files:**
- Create: `blood_experiment/robustness.py`
- Create: `tests/test_robustness.py`
- Modify: `blood_experiment/data.py`

- [ ] **Step 1: Write failing determinism and severity tests**

```python
def test_corruption_is_repeatable_for_same_seed() -> None:
    image = Image.new("RGB", (16, 16), color=(120, 80, 40))
    transform = build_corruption("brightness", severity=2, seed=42)
    assert np.array_equal(np.asarray(transform(image)), np.asarray(transform(image)))

def test_severity_zero_is_identity() -> None:
    image = Image.new("RGB", (16, 16), color=(120, 80, 40))
    assert np.array_equal(np.asarray(build_corruption("contrast", 0, 1)(image)), np.asarray(image))
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest tests/test_robustness.py -v`

Expected: FAIL because `robustness.py` is absent.

- [ ] **Step 3: Implement corruption registry**

Expose `CORRUPTION_NAMES` and `build_corruption(name, severity, seed)`. Implement brightness, contrast, saturation, hue, gamma, white balance, Gaussian blur, Gaussian noise, and JPEG round-trip. Severity 0 is identity; severities 1-3 use documented monotonically increasing fixed parameters. Apply corruptions before normalization only in evaluation transform construction. Store corruption name/severity in each prediction row.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_robustness.py tests/test_data_pipeline.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add blood_experiment/robustness.py blood_experiment/data.py tests/test_robustness.py
git commit -m "feat: add deterministic stain robustness corruptions"
```

## Task 5: Export Auxiliary Interventions and Latent Trajectories

**Files:**
- Create: `SCNODE/diagnostics/trajectory.py`
- Create: `tests/test_trajectory.py`
- Modify: `SCNODE/models/ode/scnode/scnode_resnet.py`
- Modify: `SCNODE/training/classification_trainer.py`

- [ ] **Step 1: Write failing trajectory tests**

```python
def test_pool_trajectory_separates_main_and_auxiliary_channels() -> None:
    states = torch.ones(11, 2, 5, 3, 3)
    pooled = pool_trajectory(states, augment_dim=2)
    assert pooled["main"].shape == (11, 2, 3)
    assert pooled["auxiliary"].shape == (11, 2, 2)

def test_trajectory_time_grid_has_requested_count() -> None:
    assert build_time_grid(11).tolist() == pytest.approx([index / 10 for index in range(11)])
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest tests/test_trajectory.py -v`

Expected: FAIL because the module is absent.

- [ ] **Step 3: Implement export and intervention API**

Implement `build_time_grid`, `pool_trajectory`, `export_trajectory_npz`, and `project_pca`. Add `forward_with_trajectory(x, time_points, zero_auxiliary=False)` to SCNODE. Export `time`, pooled main/auxiliary state, predicted probabilities, true label, predicted label, path, and sample index for a fixed stratified subset selected by manifest order and seed. For zero-out analysis, zero only the final ODE auxiliary channels immediately before pooling/classification, then save paired baseline/intervention predictions.

- [ ] **Step 4: Add PCA figures and tests**

Use PCA only by default so no optional UMAP dependency is introduced. Generate figures coloured separately by computational time and class. Add an optional UMAP flag that fails with an actionable install message when `umap-learn` is unavailable.

Run: `pytest tests/test_trajectory.py tests/test_scnode_config.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add SCNODE/diagnostics/trajectory.py SCNODE/models/ode/scnode/scnode_resnet.py SCNODE/training/classification_trainer.py tests/test_trajectory.py
git commit -m "feat: export SCNODE latent trajectories"
```

## Task 6: Create Review Experiment Grids and CLI

**Files:**
- Create: `SCNODE/experiments/__init__.py`
- Create: `SCNODE/experiments/configs.py`
- Create: `SCNODE/experiments/runner.py`
- Create: `SCNODE/experiments/run_review_experiment.py`
- Create: `tests/test_review_configs.py`

- [ ] **Step 1: Write failing condition-grid tests**

```python
def test_resolution_grid_has_one_condition_per_requested_pair() -> None:
    conditions = build_conditions("resolution", seeds=[42], ode_entry_sizes=[56], downsampling=["avgpool"])
    assert len(conditions) == 1
    assert conditions[0].reviewer_comment == "3"

def test_dry_run_never_creates_artifacts(tmp_path: Path) -> None:
    assert run_conditions([], output_root=tmp_path, dry_run=True) == []
    assert not tmp_path.exists()
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest tests/test_review_configs.py -v`

Expected: FAIL because the experiments package is absent.

- [ ] **Step 3: Implement condition resolution**

Define immutable `ReviewCondition` with `experiment_id`, `reviewer_comment`, `seed`, `model_name`, `ScnodeConfig`, optional corruption, and artifact directory. Build these grids:

```text
resolution:   ode_entry_size x downsampling x solver x ode_steps
augmentation: augment_dim x model_name, plus zero_auxiliary evaluation
time:         time_mode x solver x ode_steps
robustness:   model_name x corruption x severity
```

The default CLI must be deliberately small: resolution 56/28, all three downsamplers, SCNODE only, one seed for smoke tests. Full paper commands below explicitly request all conditions and seeds.

- [ ] **Step 4: Implement CLI and dry run**

```text
python -m SCNODE.experiments.run_review_experiment --experiment resolution --dry_run
python -m SCNODE.experiments.run_review_experiment --experiment all --seeds 42 123 2026 --output_root artifacts/review_experiments
```

Parse exactly: `--experiment`, `--models`, `--seeds`, `--solver`, `--ode_steps`, `--rtol`, `--atol`, `--time_mode`, `--augment_dims`, `--ode_entry_sizes`, `--downsampling`, `--trajectory_steps`, `--corruption_severities`, `--output_root`, `--dry_run`, and existing data/training parameters. Print the resolved JSON for every dry-run condition.

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_review_configs.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add SCNODE/experiments tests/test_review_configs.py
git commit -m "feat: add review experiment runner"
```

## Task 7: Aggregate Results and Produce Reviewer Tables

**Files:**
- Create: `SCNODE/experiments/summarize.py`
- Create: `tests/test_review_summary.py`
- Create: `docs/review_experiments.md`

- [ ] **Step 1: Write failing aggregation tests**

```python
def test_aggregate_rejects_incomplete_seed_groups(tmp_path: Path) -> None:
    write_summary(tmp_path / "seed_42" / "metrics" / "summary.json", macro_f1=0.8)
    with pytest.raises(ValueError, match="missing seeds"):
        aggregate_condition(tmp_path, required_seeds=[42, 123, 2026])

def test_bootstrap_difference_is_zero_for_identical_predictions() -> None:
    interval = paired_bootstrap_difference(np.array([1, 0]), np.array([1, 0]), repeats=100, seed=1)
    assert interval["mean_difference"] == 0.0
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest tests/test_review_summary.py -v`

Expected: FAIL because aggregation is absent.

- [ ] **Step 3: Implement aggregation**

Read seed-level JSON/NPZ files, validate completed seed status, calculate mean/std, and paired bootstrap macro-F1 difference CIs against the declared baseline. Generate:

```text
tables/comment_3_resolution_stability.csv
tables/comment_4_augmentation_intervention.csv
tables/comment_5_time_encoding.csv
tables/comment_11_synthetic_shift.csv
figures/comment_3_accuracy_vs_state_size.png
figures/comment_3_nfe_vs_state_size.png
figures/comment_4_pca_trajectories.png
figures/comment_10_path_smoothness.png
figures/comment_11_corruption_severity.png
rebuttal_evidence.md
```

`rebuttal_evidence.md` must contain populated values only after successful aggregation. It must state “synthetic image perturbations” for comment 11 and “model-internal computational trajectories” for comments 4/10.

- [ ] **Step 4: Document exact paper commands**

Add the following commands and expected artifacts to `docs/review_experiments.md`:

```powershell
# Comment 3: one run per condition; execute three seeds.
python -m SCNODE.experiments.run_review_experiment --experiment resolution --models SCNODE_ResNet18 --seeds 42 123 2026 --ode_entry_sizes 112 56 28 14 --downsampling maxpool avgpool stride_conv --solver rk4 --ode_steps 1 2 4 8 --output_root artifacts/review_experiments

# Comment 4 and Comment 10.
python -m SCNODE.experiments.run_review_experiment --experiment augmentation --models SCNODE_ResNet18 AnodeV2_ResNet18 --seeds 42 123 2026 --augment_dims 0 1 2 4 8 --solver rk4 --ode_steps 4 --trajectory_steps 11 --output_root artifacts/review_experiments

# Comment 5.
python -m SCNODE.experiments.run_review_experiment --experiment time --models SCNODE_ResNet18 --seeds 42 123 2026 --time_mode none concat fourier_film --solver rk4 --ode_steps 4 --output_root artifacts/review_experiments

# Comment 11: synthetic, test-only robustness.
python -m SCNODE.experiments.run_review_experiment --experiment robustness --models ResNet18 ODENet18 AnodeV2_ResNet18 SCNODE_ResNet18 --seeds 42 123 2026 --corruption_severities 0 1 2 3 --output_root artifacts/review_experiments

# Aggregate completed runs.
python -m SCNODE.experiments.summarize --input_root artifacts/review_experiments --required_seeds 42 123 2026
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_review_summary.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add SCNODE/experiments/summarize.py tests/test_review_summary.py docs/review_experiments.md
git commit -m "feat: summarize review experiment evidence"
```

## Task 8: Full Regression and Paper-Ready Run Sequence

**Files:**
- Modify only if failures require focused fixes from Tasks 1-7.

- [ ] **Step 1: Run all automated tests**

Run: `pytest -q`

Expected: PASS with no skipped test masking a required review feature.

- [ ] **Step 2: Execute a one-epoch smoke condition**

```powershell
python -m SCNODE.experiments.run_review_experiment --experiment resolution --models SCNODE_ResNet18 --seeds 42 --ode_entry_sizes 56 --downsampling maxpool --solver rk4 --ode_steps 2 --num_epochs 1 --num_workers 0 --generate_cam false --output_root artifacts/review_smoke
```

Expected: a seed directory containing resolved config, checkpoint, prediction NPZ, epoch metrics, diagnostic JSON, and a completed status file.

- [ ] **Step 3: Validate result integrity before long runs**

Run: `python -m SCNODE.experiments.summarize --input_root artifacts/review_smoke --required_seeds 42`

Expected: generated table and `rebuttal_evidence.md`; no cross-domain or biological-time wording.

- [ ] **Step 4: Execute the four paper command groups from Task 7**

Run them serially or on separate GPUs, retaining the same manifest path and three seeds. Do not inspect test metrics to alter hyperparameters after a condition has started.

- [ ] **Step 5: Produce rebuttal claims from actual outcomes only**

Use these conditional response patterns:

```text
Comment 3: “Across [state sizes], the best trade-off occurred at [size]. The effect is empirical for this architecture: [Macro-F1 CI], [NFE], and [endpoint error] changed together; we do not posit a universal dimensional threshold.”
Comment 4: “Auxiliary-channel removal changed Macro-F1 by [CI], and the exported trajectories show [measured association]. We therefore describe SAM as generic latent augmentation rather than biologically structured variables.”
Comment 5: “With a multi-step solver, [time mode] changed Macro-F1 by [CI] versus autonomous dynamics. Numerical time is computational depth; the static dataset does not identify irregular biological maturation rates.”
Comment 10: “The paths have [path-length/curvature/solver-agreement] values. These are model-internal trajectories, not observed cell differentiation trajectories.”
Comment 11: “Under controlled synthetic colour/acquisition perturbations, SCNODE retained [Macro-F1] relative to [baseline]. This evaluates simulated robustness only; no inter-laboratory dataset was available.”
```

- [ ] **Step 6: Commit focused final fixes only**

```bash
git add <files changed by regression fixes>
git commit -m "fix: validate SCNODE review experiment workflow"
```

## Plan Self-Review

- Coverage: Tasks 1-3 answer numerical stability; Task 5 answers auxiliary and trajectory requests; Task 4 and Task 6 answer test-time shift; Task 7 produces reviewer-specific evidence; Task 8 prevents fabricated results.
- Scope: The plan deliberately excludes claims requiring unavailable longitudinal or site-labelled data.
- Consistency: Every CLI name in the specification appears in Task 6, every generated result has a responsible owner, and every implementation task starts with a failing test.
