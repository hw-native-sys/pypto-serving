from pathlib import Path


CORE_DIR = Path(__file__).parents[4] / "pypto_serving" / "serving" / "pd"
ROUTER_DIR = Path(__file__).parents[4] / "pypto_serving" / "router"


def test_generic_pd_core_has_no_model_specific_imports_or_layout_names() -> None:
    forbidden = (
        "model.deepseek",
        "DSV4Registry",
        "hca_cmp",
        "csa_state",
        "idx_scale",
        "PD_DSPARK",
        "PD_PHYSICAL_REGIONS",
    )
    sources = tuple(
        source
        for directory in (CORE_DIR, ROUTER_DIR)
        for source in directory.glob("*.py")
        if not source.name.startswith("._")
    )
    for source in sources:
        text = source.read_text(encoding="utf-8")
        assert not any(value in text for value in forbidden), source
