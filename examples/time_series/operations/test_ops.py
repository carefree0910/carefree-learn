import os
import random
import cflearn

import numpy as np

from typing import Any
from datetime import datetime, timedelta
from cfdata.tabular import TimeSeriesConfig


CI = True

num_case = 10
num_sample = 100 if CI else 10000
num_history = 5
sf_name = f"sum_{num_history}"
pf_name = f"prod_{num_history}"
file_folder = os.path.dirname(__file__)
sf_file = os.path.join(file_folder, f"{sf_name}.csv")
pf_file = os.path.join(file_folder, f"{pf_name}.csv")
sf_te_file = os.path.join(file_folder, f"{sf_name}_te.csv")
pf_te_file = os.path.join(file_folder, f"{pf_name}_te.csv")

kwargs = {"fixed_epoch": 3} if CI else {}


def make_datasets() -> None:
    def write(num_id: int, num_line: int, sum_f: Any, prod_f: Any) -> None:
        header = "id,Date,Value,Label\n"
        sum_f.write(header)
        prod_f.write(header)
        begin = datetime(2020, 1, 1)
        for i in range(num_id):
            last = None
            s_label, p_label = 0.0, 1.0
            id_ = f"case{i}"
            values = []
            for j in range(num_line):
                delta = timedelta(j)
                current = (begin + delta).strftime("%Y-%m-%d")
                value = 2 * random.random() - 1
                prefix = f"{id_},{current},{value}"
                if last is not None:
                    s_label -= last
                    p_label /= last
                s_label += value
                p_label *= value
                sum_f.write(f"{prefix},{s_label}\n")
                prod_f.write(f"{prefix},{p_label}\n")
                values.append(value)
                if j >= num_history - 1:
                    last = values[j - num_history + 1]

    with open(sf_file, "w") as sf, open(pf_file, "w") as pf:
        write(num_case, num_sample, sf, pf)
    with open(sf_te_file, "w") as sf_te, open(pf_te_file, "w") as pf_te:
        write(num_case, num_history + 1, sf_te, pf_te)


def test_ops() -> None:

    make_datasets()

    ts_config = TimeSeriesConfig("id", "Date")
    aggregation_config = {"num_history": num_history}
    data_config = {
        "default_numerical_process": "identical",
        "label_process_method": "identical",
    }

    ops = ["sum", "prod", "sum", "prod"]
    models = ["linear", "rnn", "transformer", "transformer"]

    for op, model in zip(ops, models):
        task = f"{op}_{num_history}"

        m = cflearn.make(
            model,
            ts_config=ts_config,
            aggregation_config=aggregation_config,
            data_config=data_config,
            **kwargs,  # type: ignore
        )

        tr_file = os.path.join(file_folder, f"{task}.csv")
        te_file = os.path.join(file_folder, f"{task}_te.csv")
        m.fit(tr_file)
        predictions = m.predict(te_file, contains_labels=True)
        labels = m.data.read_file(te_file)[1].to_numpy().astype(np.float32)  # type: ignore
        labels = labels.reshape([-1, num_history + 1])[..., -2:].reshape([-1, 1])
        print(np.hstack([predictions, labels]))  # type: ignore


if __name__ == "__main__":
    test_ops()
