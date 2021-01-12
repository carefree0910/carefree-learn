import os
import mlflow
import cflearn
import argparse

import numpy as np

from cflearn.configs import _parse_config
from cflearn.configs import Elements
from cflearn.configs import Environment
from cflearn.misc.toolkit import parse_args
from cflearn.misc.toolkit import parse_path
from mlflow.tracking.fluent import _active_experiment_id

try:
    import deepspeed
except:
    deepspeed = None


if __name__ == "__main__":
    if deepspeed is None:
        raise ValueError("deepspeed is not installed")
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    deepspeed.add_config_arguments(parser)
    parser.add_argument("--model")
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
    environment = Environment.from_elements(Elements.make(config), False)
    logging_folder = environment.logging_folder
    if os.path.abspath(logging_folder) != logging_folder:
        logging_folder = parse_path(logging_folder, root_dir)
    config["logging_folder"] = logging_folder
    cflearn.impute_deepspeed_args(args, config, environment.trainer_config)
    if args.deepspeed_config is None:
        config.setdefault("log_pipeline_to_artifacts", True)
        mlflow_config = config.setdefault("mlflow_config", {})
        if root_dir is not None:
            mlflow_config["tracking_folder"] = root_dir
        mlflow_config["task_name"] = mlflow.get_experiment(_active_experiment_id).name

    model_saving_folder = config.pop("model_saving_folder", None)
    m = cflearn.make(args.model, **config).fit(x, x_cv=x_cv)
    if model_saving_folder is not None and m.is_rank_0:
        m.save(os.path.join(model_saving_folder, "pipeline"))
