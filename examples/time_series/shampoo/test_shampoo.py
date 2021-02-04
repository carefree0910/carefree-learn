import os
import cflearn

from cfdata.tabular import TimeSeriesConfig


CI = True

file_folder = os.path.dirname(__file__)
kwargs = {"fixed_epoch": 3} if CI else {}


def test_shampoo() -> None:
    ts_config = TimeSeriesConfig("ID", "Month")
    aggregation_config = {"num_history": 2}
    m = cflearn.make(
        ts_config=ts_config,
        aggregation_config=aggregation_config,
        **kwargs,  # type: ignore
    )
    m.fit(os.path.join(file_folder, "data.csv"))


if __name__ == "__main__":
    test_shampoo()
