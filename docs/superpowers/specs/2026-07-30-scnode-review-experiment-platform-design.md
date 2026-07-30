# SCNODE Review Experiment Platform Design

**Goal:** Add a reproducible experiment and diagnostics platform that produces the evidence required to answer reviewer comments 3, 4, 5, 10, and 11 without overstating what a static image-classification dataset can establish.

## Evidence Boundary

The BM cytomorphology dataset contains static images and class labels but no longitudinal cell trajectories, biological timestamps, laboratory identifiers, scanner identifiers, stain batches, or external cohort. The platform may measure model-internal feature paths and robustness to synthetic image perturbations. It must not label those paths as observed biological maturation or claim real cross-site generalization.

## Correctness Prerequisite

The active SCNODE model uses a two-point integration interval with the fixed Euler solver. The new ODE interface must expose an adaptive solver and a multi-step fixed solver configuration. All ablations use the same configured solver. Function-evaluation count, state norms, non-finite values, and gradient norms are captured for every run.

## Architecture

### Model Configuration

`SCNODE/models/ode/scnode/scnode_resnet.py` becomes a thin public factory around focused modules:

- `config.py`: validated model and solver dataclasses.
- `blocks.py`: time-conditioning layers, ODE function, augmentation, and ODE block.
- `resnet.py`: SCNODE stem and residual stages with configurable downsampler.

Supported time modes are `none`, `concat`, and `fourier_film`. Supported downsamplers are `maxpool`, `avgpool`, and `stride_conv`; all produce an identical spatial shape. Auxiliary dimensions are a configurable integer and remain generic latent dimensions, not biological groups.

### Experiment Runners

`SCNODE/experiments/` owns declarative experiment grids and one runner per reviewer concern. Each runner creates a self-contained directory under `artifacts/review_experiments/<experiment_id>/<seed>/` and writes a resolved JSON configuration before training.

- `resolution_stability.py`: resolution/downsampler/solver grid.
- `augmentation_trajectory.py`: auxiliary-dimension ablation, zero-out intervention, trajectory export.
- `time_encoding.py`: autonomous, scalar-concatenation, and Fourier-FiLM time modes.
- `synthetic_shift.py`: test-only colour and acquisition corruption severity grid.
- `summarize.py`: seed aggregation, confidence intervals, tables, and figures.

### Diagnostics and Robustness

`SCNODE/diagnostics/ode_metrics.py` collects per-block NFEs, state norms, relative endpoint error, non-finite counts, and gradient norms. `SCNODE/diagnostics/trajectory.py` exports evenly sampled ODE states, pooled original/auxiliary features, and metadata. `blood_experiment/robustness.py` provides deterministic image perturbations that are only applied to validation/test datasets.

## Shared Experimental Protocol

- Use the existing fixed manifest split, three seeds (`42, 123, 2026`), identical optimizer, epoch count, early-stopping rule, and preprocessing across ablations.
- Select the checkpoint by validation macro-F1; evaluate the test set exactly once per seed and condition.
- Report mean +/- standard deviation and paired bootstrap 95% confidence intervals over test predictions. Use macro-F1 as the primary endpoint, plus balanced accuracy, MCC, classwise recall, parameter count, wall-clock time, peak memory, and NFE.
- Always include ResNet18, ODENet18, vanilla ANODE, and SCNODE when making a comparative SCNODE claim. Parameter-match ablations where practical.

## Reviewer Evidence Matrix

### Comment 3: Input Resolution and ODE Stability

Hold the image input at 224. Vary the pre-ODE state grid (`112`, `56`, `28`, `14`) and downsampler (`maxpool`, `avgpool`, `stride_conv`). For each condition compare `euler` at 1/2/4/8 steps and `dopri5` at fixed tolerances. Report classification performance, NFE, memory, state/gradient norms, non-finite rate, and endpoint error relative to a tight-tolerance reference. Define any threshold as an empirical configuration-specific inflection point, never a universal dimensional threshold.

### Comment 4: SAM and Topology

Compare `augment_dim` 0/1/2/4/8 and vanilla ANODE at matched total width. Apply a test-time zero-out intervention to the auxiliary channels. Export eleven solver states for a stratified held-out subset; create PCA/UMAP plots coloured separately by computational time, class, and any pre-specified coarse label mapping. Report linear-probe classification of pooled auxiliary features and silhouette scores. State that these are latent associations, not biological semantic proof.

### Comment 5: Time Conditioning

Compare `none`, `concat`, and `fourier_film`, using multiple numerical integration steps for every condition. Compare linear numerical time to a monotone learned reparameterization only as a computational-depth ablation. Do not claim recovery of irregular biological maturation rates because no biological time label exists.

### Comment 10: Biological Interpretability

Use trajectory exports to quantify smoothness (adjacent-state displacement, total path length, curvature) and solver agreement. Visualize representative correctly classified and confused samples. Treat trajectories as model-internal continuous-depth paths. Include a limitation paragraph stating that the static dataset cannot validate a cell's actual differentiation trajectory or monotonic biological maturation.

### Comment 11: Domain Shift and Stain Variation

Apply deterministic, test-only brightness, contrast, saturation, hue, gamma, white-balance, blur, Gaussian-noise, and JPEG perturbations at severity 0-3. Compare the same four model families, with mean corruption error and per-corruption macro-F1. Explicitly label the study as simulated robustness, not domain adaptation or real inter-laboratory generalization.

## CLI Contract

The existing `run_bm_experiment.py` retains simple single-model training. A new `run_review_experiment.py` exposes:

```text
--experiment {resolution,augmentation,time,robustness,all}
--models SCNODE_ResNet18 ResNet18 ODENet18 AnodeV2_ResNet18
--seeds 42 123 2026
--solver {euler,rk4,dopri5}
--ode_steps 1 2 4 8
--rtol 1e-3 --atol 1e-3
--time_mode {none,concat,fourier_film}
--augment_dims 0 1 2 4 8
--ode_entry_sizes 112 56 28 14
--downsampling {maxpool,avgpool,stride_conv}
--trajectory_steps 11
--corruption_severities 0 1 2 3
--output_root artifacts/review_experiments
```

## Acceptance Criteria

1. A dry-run prints every resolved condition and creates no training jobs.
2. Every completed condition contains `config.json`, per-epoch metrics, per-sample predictions, ODE diagnostics, and a checkpoint.
3. Aggregation rejects incomplete seed groups and outputs CSV, Markdown, and publication-ready figures with uncertainty.
4. Tests cover configuration validation, tensor shapes for all model variants, deterministic corruptions, metric aggregation, trajectory export, and intervention behavior.
5. The documentation maps every generated table and figure to the exact reviewer comment it supports and states the evidence boundary.
