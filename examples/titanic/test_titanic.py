# This is a solution for https://www.kaggle.com/c/titanic
# Uncomment the `model_config` parts can achieve better performances!

import os
import cflearn

from cfdata.tabular import *

file_folder = os.path.dirname(__file__)


def _hpo_core(train_file):
    data_config = {"label_name": "Survived"}
    # model_config = {"default_encoding_configs": {"embedding_std": 1.0}}
    hpo = cflearn.tune_with(
        train_file,
        model="tree_dnn",
        temp_folder="__test_titanic1__",
        task_type=TaskTypes.CLASSIFICATION,
        # model_config=model_config,
        data_config=data_config,
        num_parallel=0,
    )
    results = cflearn.repeat_with(
        train_file,
        **hpo.best_param,
        models="tree_dnn",
        temp_folder="__test_titanic2__",
        num_repeat=10,
        num_jobs=0,
        data_config=data_config,
    )
    ensemble = cflearn.ensemble(results.patterns["tree_dnn"])
    return results.data, ensemble


def _adaboost_core(train_file):
    config = {
        "data_config": {"label_name": "Survived"},
        # "model_config": {"default_encoding_configs": {"embedding_std": 1.0}},
    }
    ensemble = cflearn.Ensemble(TaskTypes.CLASSIFICATION, config)
    results = ensemble.adaboost(train_file)
    return results.data, results.pattern


def _test(name, _core):
    train_file = os.path.join(file_folder, "train.csv")
    test_file = os.path.join(file_folder, "test.csv")
    data, pattern = _core(train_file)
    predictions = pattern.predict(test_file).ravel()
    x_te, _ = data.read_file(test_file, contains_labels=False)
    id_list = DataTuple.with_transpose(x_te, None).xT[0]
    # Score : achieved ~0.79
    with open(f"submissions_{name}.csv", "w") as f:
        f.write("PassengerId,Survived\n")
        for test_id, prediction in zip(id_list, predictions):
            f.write(f"{test_id},{prediction}\n")


def test_hpo():
    _test("hpo", _hpo_core)


def test_adaboost():
    _test("adaboost", _adaboost_core)


if __name__ == "__main__":
    test_adaboost()
    test_hpo()
