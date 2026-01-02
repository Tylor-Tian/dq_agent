import pandas as pd

from dq_agent.config import Config, ColumnConfig, DatasetConfig
from dq_agent.contract import validate_contract


def test_contract_missing_column():
    cfg = Config(
        dataset=DatasetConfig(primary_key=["order_id"]),
        columns={
            "order_id": ColumnConfig(type="string", required=True),
            "amount": ColumnConfig(type="float", required=True),
        },
    )
    df = pd.DataFrame({"order_id": ["1", "2"]})
    issues = validate_contract(df, cfg)
    messages = [issue.message for issue in issues]
    assert any("amount" in message for message in messages)
