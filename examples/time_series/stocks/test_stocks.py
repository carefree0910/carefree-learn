import os
import cflearn

from cfdata.tabular import TimeSeriesConfig
from cfdata.tabular import TimeSeriesModifier


CI = True

file_folder = os.path.dirname(__file__)
kwargs = {"fixed_epoch": 3} if CI else {}


def test_stocks() -> None:
    src_file = os.path.join(file_folder, "data.csv")
    tgt_file = os.path.join(file_folder, "new_data.csv")
    ts_config = TimeSeriesConfig("id", "Date")
    aggregation_config = {"num_history": 5}
    modifier = TimeSeriesModifier(src_file, "ts_reg", ts_config=ts_config)
    modifier.pad_labels(lambda batch: batch[..., -1, 3:4], offset=1)
    modifier.export_to(tgt_file)
    m = cflearn.make(
        ts_config=ts_config,
        aggregation_config=aggregation_config,
        **kwargs,  # type: ignore
    )
    m.fit(os.path.join(file_folder, tgt_file))


if __name__ == "__main__":
    test_stocks()
