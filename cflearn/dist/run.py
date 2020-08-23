import os
import argparse

import numpy as np

from cftool.misc import Saving

import cflearn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_folder")
    args = parser.parse_args()
    config = Saving.load_dict("config", args.config_folder)
    sample_weights = config.pop("sample_weights", None)
    logging_folder = config["logging_folder"]
    data_folder = config.get("data_folder", logging_folder)
    keys = ["x", "y", "x_cv", "y_cv"]
    data_list = list(map(config.pop, keys, 4 * [None]))
    if data_list[0] is None:
        for i, key in enumerate(keys):
            data_file = os.path.join(data_folder, f"{key}.npy")
            if os.path.isfile(data_file):
                data_list[i] = np.load(data_file)
    trains_config = config.pop("trains_config", None)
    m = cflearn.make(**config)
    m.trains(*data_list, sample_weights=sample_weights, trains_config=trains_config)
    cflearn.save(m, saving_folder=logging_folder)
