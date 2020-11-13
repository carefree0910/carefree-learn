import os
import cflearn

from cfdata.tabular import TimeSeriesConfig
from cfdata.tabular import TimeSeriesModifier


CI = True

file_folder = os.path.dirname(__file__)
kwargs = {"fixed_epoch": 3} if CI else {}


def test_shampoo() -> None:
    src_file = os.path.join(file_folder, "data.csv")
    tgt_file = os.path.join(file_folder, "new_data.csv")
    ts_config = TimeSeriesConfig(TimeSeriesModifier.id_name, "Month")
    modifier = TimeSeriesModifier(src_file, "ts_reg")
    modifier.pad_id()
    modifier.pad_labels(
        lambda batch: batch[..., -1, :],
        offset=1,
        ts_config=ts_config,
    )
    modifier.export_to(tgt_file)
    aggregation_config = {"num_history": 2}
    m = cflearn.make(
        ts_config=ts_config,
        aggregation_config=aggregation_config,
        **kwargs,  # type: ignore
    )
    m.fit(os.path.join(file_folder, tgt_file))


if __name__ == "__main__":
    test_shampoo()
