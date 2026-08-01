# SCNODE Reviewer-Response Experiments

All paper runs use the same split manifest (`--split_seed 42`), select a
checkpoint by validation macro-F1, and access the test split once after model
selection. Use the three training seeds `42 123 2026`; do not tune parameters
from any test result. `--dry_run` prints the resolved condition list without
creating artifacts.

Before a non-dry run, activate the training environment and verify the ODE
dependency is available:

```powershell
conda run -n deep_learning python -c "import torch, torchdiffeq; print(torch.__version__)"
```

## Smoke check (all comments)

```powershell
conda run -n deep_learning python -m SCNODE.experiments.run_review_experiment --experiment resolution --models SCNODE_ResNet18 --seeds 42 --ode_entry_sizes 56 --downsampling maxpool --num_epochs 1 --batch_size 4 --test_batch_size 4 --num_workers 0 --cpu --dry_run
```

## Comment 3 — resolution, downsampler, and numerical stability

The main comparison holds image input at 224 and changes only the spatial grid
at the ODE entry. It reports macro-F1, memory/time, NFE, state/gradient norms,
and non-finite batches. State-size effects are architecture-specific empirical
results, not a universal dimensional threshold.

```powershell
conda run -n deep_learning python -m SCNODE.experiments.run_review_experiment --experiment resolution --models SCNODE_ResNet18 --seeds 42 123 2026 --ode_entry_sizes 112 56 28 14 --downsampling maxpool avgpool stride_conv --solver rk4 --ode_steps 4 --split_seed 42 --output_root artifacts/review_experiments
```

Run the solver sensitivity ablation separately for Euler and adaptive Dopri5:

```powershell
conda run -n deep_learning python -m SCNODE.experiments.run_review_experiment --experiment resolution --models SCNODE_ResNet18 --seeds 42 123 2026 --ode_entry_sizes 112 56 28 14 --downsampling maxpool avgpool stride_conv --solver euler --ode_steps 1 --split_seed 42 --output_root artifacts/review_experiments
conda run -n deep_learning python -m SCNODE.experiments.run_review_experiment --experiment resolution --models SCNODE_ResNet18 --seeds 42 123 2026 --ode_entry_sizes 112 56 28 14 --downsampling maxpool avgpool stride_conv --solver dopri5 --rtol 1e-5 --atol 1e-5 --split_seed 42 --output_root artifacts/review_experiments
```

## Comments 4 and 10 — SAM ablation and computational trajectories

This exports per-ODE-block trajectories for a fixed stratified test subset. The
paths are internal computational trajectories, not observed cell maturation.

```powershell
conda run -n deep_learning python -m SCNODE.experiments.run_review_experiment --experiment augmentation --models SCNODE_ResNet18 AnodeV2_ResNet18 --seeds 42 123 2026 --augment_dims 0 1 2 4 8 --solver rk4 --ode_steps 4 --trajectory_steps 11 --trajectory_samples 21 --split_seed 42 --output_root artifacts/review_experiments
```

## Comment 5 — temporal conditioning

`none` is autonomous dynamics, `concat` is the submitted TConv design, and
`fourier_film` is a learned non-autonomous alternative. The integration time is
normalised computational depth rather than biological time.

```powershell
conda run -n deep_learning python -m SCNODE.experiments.run_review_experiment --experiment time --models SCNODE_ResNet18 --seeds 42 123 2026 --time_modes none concat fourier_film --solver rk4 --ode_steps 4 --split_seed 42 --output_root artifacts/review_experiments
```

## Comment 11 — simulated stain/acquisition shifts

These are deterministic test-only image corruptions. They establish synthetic
robustness, not cross-laboratory generalisation or domain adaptation.

```powershell
conda run -n deep_learning python -m SCNODE.experiments.run_review_experiment --experiment robustness --models ResNet18 ODENet18 AnodeV2_ResNet18 SCNODE_ResNet18 --seeds 42 123 2026 --corruptions brightness contrast saturation hue gamma white_balance gaussian_blur gaussian_noise jpeg --corruption_severities 0 1 2 3 --split_seed 42 --output_root artifacts/review_experiments
```

## Aggregate only complete seed groups

```powershell
conda run -n deep_learning python -m SCNODE.experiments.summarize --input_root artifacts/review_experiments --required_seeds 42 123 2026
```

Each condition writes its resolved configuration and status below
`artifacts/review_experiments/<experiment>/<model>/<condition-hash>/seed_<seed>/`.
