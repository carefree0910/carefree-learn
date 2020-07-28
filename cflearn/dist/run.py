import os
import json
import argparse

import numpy as np

import cflearn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file")
    args = parser.parse_args()
    with open(args.config_file, "r") as f:
        config = json.load(f)
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
    m.trains(*data_list, trains_config=trains_config)
    cflearn.save(m, saving_folder=logging_folder)
