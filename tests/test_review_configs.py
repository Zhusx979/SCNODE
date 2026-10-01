from SCNODE.experiments.configs import build_conditions


def test_resolution_grid_assigns_reviewer_comment_three() -> None:
    conditions = build_conditions(
        "resolution",
        models=["SCNODE_ResNet18"],
        seeds=[42],
        ode_entry_sizes=[56],
        downsampling=["avgpool"],
    )

    assert len(conditions) == 1
    assert conditions[0].reviewer_comments == ("3",)
    assert conditions[0].scnode_config.ode_entry_size == 56


def test_robustness_grid_assigns_reviewer_comment_eleven() -> None:
    conditions = build_conditions(
        "robustness",
        models=["SCNODE_ResNet18"],
        seeds=[42],
        corruptions=["brightness"],
        corruption_severities=[2],
    )

    assert conditions[0].reviewer_comments == ("11",)
    assert conditions[0].corruption == "brightness"


def test_robustness_grid_creates_one_shared_clean_condition() -> None:
    conditions = build_conditions(
        "robustness", models=["SCNODE_ResNet18"], seeds=[42],
        corruptions=["brightness", "gaussian_noise"], corruption_severities=[0, 3],
    )

    assert len(conditions) == 3
    assert sum(condition.corruption is None for condition in conditions) == 1


def test_ablation_grid_exposes_independent_component_switches() -> None:
    conditions = build_conditions("component_ablation", models=["SCNODE_ResNet18"], seeds=[42])
    assert [condition.experiment for condition in conditions] == [
        "bn_only", "sam_only", "tconv_only", "twbn_only", "full_scnnode"
    ]
    assert [(c.scnode_config.use_bn, c.scnode_config.use_sam,
             c.scnode_config.use_tconv, c.scnode_config.use_twbn) for c in conditions] == [
        (True, False, False, False),
        (False, True, False, False),
        (False, False, True, False),
        (False, False, False, True),
        (True, True, True, True),
    ]


def test_ablation_grid_exposes_paper_rows() -> None:
    conditions = build_conditions("ablation", models=["SCNODE_ResNet18"], seeds=[42])
    names = {condition.experiment for condition in conditions}
    assert {"A_entry_28", "A_avgpool", "B_aux_4", "C_autonomous",
            "D_window_7", "D_grid_17", "D_quantile"} <= names
