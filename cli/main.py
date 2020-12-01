import mlflow
import cflearn
import argparse

from cflearn.configs import _parse_config
from cflearn.misc.toolkit import parse_args
from cflearn.misc.toolkit import parse_path
from mlflow.tracking.fluent import _active_experiment_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    x, x_cv, saving_folder = map(parse_path, [x, x_cv, saving_folder], 3 * [root_dir])
    config = _parse_config(args.config)
    config.setdefault("log_pipeline_to_artifacts", True)
    mlflow_config = config.setdefault("mlflow_config", {})
    mlflow_config["tracking_folder"] = root_dir
    mlflow_config["task_name"] = mlflow.get_experiment(_active_experiment_id).name
    m = cflearn.make(args.model, **config).fit(x, x_cv=x_cv)
