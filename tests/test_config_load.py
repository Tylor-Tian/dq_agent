from pathlib import Path

from dq_agent.config import load_config


def test_config_load():
    config_path = Path(__file__).parents[1] / "dq_agent" / "resources" / "demo_rules.yml"
    cfg = load_config(config_path)
    assert cfg.dataset.name == "demo_orders"
    assert "order_id" in cfg.columns
