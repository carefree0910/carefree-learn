import os
import cflearn
import argparse

import numpy as np

from cflearn.misc.toolkit import parse_args
from cflearn.misc.toolkit import parse_path
from cflearn.misc.toolkit import _parse_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--root_dir", default=None)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--valid_file", default=None)
    args = parse_args(parser.parse_args())
    root_dir = args.root_dir
    train, valid = args.train_file, args.valid_file
    train, valid, config_path = map(
        parse_path,
        [train, valid, args.config],
        3 * [root_dir],
    )
    if not train.endswith(".npy"):
        x, x_cv = train, valid
        y = y_cv = None
    else:
        train = np.load(train)
        x, y = np.split(train, [-1], axis=1)
        if valid is None:
            x_cv = y_cv = None
        else:
            if not valid.endswith(".npy"):
                msg = f"train_file is a numpy array but valid_file ({valid}) is not"
                raise ValueError(msg)
            valid = np.load(valid)
            x_cv, y_cv = np.split(valid, [-1], axis=1)
    config = _parse_config(config_path)
    workplace = config.get("workplace")
    if workplace is None:
        workplace = "_logs"
    if os.path.abspath(workplace) != workplace:
        workplace = parse_path(workplace, root_dir)
    config["workplace"] = workplace
    model_saving_folder = config.pop("model_saving_folder", None)
    m = cflearn.make(args.pipeline, config=config).fit(x, y, x_cv, y_cv)
    if model_saving_folder is not None and m.is_rank_0:
        m.save(os.path.join(model_saving_folder, "pipeline"))
