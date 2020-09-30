import os
import cflearn

from cfdata.tabular import *

file_folder = os.path.dirname(__file__)


def test_shampoo():
    src_file = os.path.join(file_folder, "data.csv")
    tgt_file = os.path.join(file_folder, "new_data.csv")
    ts_config = TimeSeriesConfig(TimeSeriesModifier.id_name, '"Month"')
    modifier = TimeSeriesModifier(
        src_file,
        TaskTypes.TIME_SERIES_REG,
    )
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
    )
    m.fit(os.path.join(file_folder, tgt_file))


if __name__ == "__main__":
    test_shampoo()
