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
