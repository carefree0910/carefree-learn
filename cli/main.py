import os
import json
import mlflow
import cflearn
import argparse

from cftool.misc import lock_manager
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


def _impute_deepspeed() -> None:
    if args.deepspeed_config is not None:
        config["use_tqdm"] = False
        config["trigger_logging"] = True
        config["ds_args"] = args
        with open(args.deepspeed_config, "r") as f:
            ds_config = json.load(f)
        ds_config_changed = False
        if environment.trainer_config["use_amp"]:
            ds_config_changed = True
            ds_config["fp16"] = {
                "enabled": True,
                "loss_scale": 0,
                "initial_scale_power": 32,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1,
            }
        clip_norm = environment.trainer_config["clip_norm"]
        if clip_norm > 0.0:
            ds_config_changed = True
            ds_config["gradient_clipping"] = clip_norm
        if ds_config_changed:
            folder, file = os.path.split(args.deepspeed_config)
            name, ext = os.path.splitext(file)
            new_file = f"_cf_{name}{ext}"
            new_path = new_file if not folder else os.path.join(folder, new_file)
            with lock_manager(folder or "./", new_file):
                with open(new_path, "w") as f:
                    json.dump(ds_config, f)
            args.deepspeed_config = new_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    deepspeed.add_config_arguments(parser)
    parser.add_argument("--model")
    parser.add_argument("--config", default=None)
    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--valid_file", default=None)
    parser.add_argument("--saving_folder", default=None)
    args = parse_args(parser.parse_args())
    root_dir = args.root_dir
    saving_folder = args.saving_folder
    x, x_cv = args.train_file, args.valid_file
    x, x_cv, config_path, saving_folder = map(
        parse_path,
        [x, x_cv, args.config, saving_folder],
        4 * [root_dir],
    )
    config = _parse_config(config_path)
    environment = Environment.from_elements(Elements.make(config), False)
    logging_folder = environment.logging_folder
    if os.path.abspath(logging_folder) != logging_folder:
        logging_folder = parse_path(logging_folder, root_dir)
    config["logging_folder"] = logging_folder
    if deepspeed is None:
        raise ValueError("deepspeed is not supported")
    _impute_deepspeed()
    if args.deepspeed_config is None:
        config.setdefault("log_pipeline_to_artifacts", True)
        mlflow_config = config.setdefault("mlflow_config", {})
        mlflow_config["tracking_folder"] = root_dir
        mlflow_config["task_name"] = mlflow.get_experiment(_active_experiment_id).name
    m = cflearn.make(args.model, **config).fit(x, x_cv=x_cv)
