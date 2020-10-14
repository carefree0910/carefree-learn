# This is a solution for https://www.kaggle.com/c/titanic
# Set `CI = False` can achieve better performances!

import os
import cflearn

from cfdata.tabular import *

model = "fcnn"
file_folder = os.path.dirname(__file__)

CI = True


def _hpo_core(train_file):
    extra_config = {"data_config": {"label_name": "Survived"}}
    if CI:
        extra_config.update({"min_epoch": 1, "num_epoch": 2, "max_epoch": 4})
    hpo_temp_folder = "__test_titanic_hpo__"
    result = cflearn.tune_with(
        train_file,
        model=model,
        temp_folder=os.path.join(hpo_temp_folder, "__tune__"),
        task_type=TaskTypes.CLASSIFICATION,
        extra_config=extra_config,
        num_parallel=0,
        num_repeat=2 if CI else 5,
        num_search=5 if CI else 10,
    )
    results = cflearn.repeat_with(
        train_file,
        **result.best_param,
        models=model,
        temp_folder=os.path.join(hpo_temp_folder, "__repeat__"),
        num_repeat=2 if CI else 10,
        num_jobs=0,
    )
    ensemble = cflearn.ensemble(results.patterns[model])
    return results.data, ensemble


def _optuna_core(train_file):
    extra_config = {"data_config": {"label_name": "Survived"}}
    if CI:
        extra_config.update({"min_epoch": 1, "num_epoch": 2, "max_epoch": 4})
    opt = cflearn.Auto(TaskTypes.CLASSIFICATION).fit(
        train_file,
        model=model,
        temp_folder="__test_titanic_optuna__",
        extra_config=extra_config,
        num_final_repeat=2 if CI else 10,
        num_repeat=2 if CI else 5,
        num_trial=4 if CI else 100,
        num_jobs=2,
    )
    return opt.data, opt.pattern


def _adaboost_core(train_file):
    config = {"data_config": {"label_name": "Survived"}}
    if CI:
        config.update({"min_epoch": 1, "num_epoch": 2, "max_epoch": 4})
    ensemble = cflearn.Ensemble(TaskTypes.CLASSIFICATION, config)
    results = ensemble.adaboost(train_file, model=model)
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
    _test("optuna", _optuna_core)


def test_adaboost():
    _test("adaboost", _adaboost_core)


if __name__ == "__main__":
    test_adaboost()
    test_hpo()
