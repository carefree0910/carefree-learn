# This is a solution for https://www.kaggle.com/c/titanic
# Set `CI = False` can achieve better performances!

import os
import cflearn

from typing import Any
from typing import Dict
from typing import Tuple
from typing import Callable

from cftool.ml import pattern_type
from cftool.misc import shallow_copy_dict
from cfdata.tabular import DataTuple
from cfdata.tabular import TabularData


model = "tree_dnn"
file_folder = os.path.dirname(__file__)

CI = True


def _hpo_core(train_file: str) -> Tuple[TabularData, pattern_type]:
    extra_config: Dict[str, Any] = {"data_config": {"label_name": "Survived"}}
    if CI:
        extra_config.update({"min_epoch": 1, "num_epoch": 2, "max_epoch": 4})
    hpo_temp_folder = "__test_titanic_hpo__"
    tune_result = cflearn.tune_with(
        train_file,
        model=model,
        task_type="clf",
        temp_folder=os.path.join(hpo_temp_folder, "__tune__"),
        extra_config=extra_config,
        num_parallel=0,
        num_repeat=2 if CI else 5,
        num_search=5 if CI else 10,
    )
    repeat_config = shallow_copy_dict(tune_result.best_param)
    repeat_config.update(
        {
            "models": model,
            "temp_folder": os.path.join(hpo_temp_folder, "__repeat__"),
            "num_repeat": 2 if CI else 10,
            "num_jobs": 0,
        }
    )
    repeat_result = cflearn.repeat_with(train_file, **repeat_config)
    patterns = repeat_result.patterns
    assert patterns is not None
    ensemble = cflearn.ensemble(patterns[model])
    return repeat_result.data, ensemble


def _optuna_core(train_file: str) -> Tuple[TabularData, pattern_type]:
    extra_config: Dict[str, Any] = {"data_config": {"label_name": "Survived"}}
    if CI:
        extra_config.update({"min_epoch": 1, "num_epoch": 2, "max_epoch": 4})
    auto = cflearn.Auto("clf", model=model)
    auto.fit(
        train_file,
        temp_folder="__test_titanic_optuna__",
        extra_config=extra_config,
        num_final_repeat=2 if CI else 10,
        num_repeat=2 if CI else 5,
        num_trial=4 if CI else 100,
    )
    return auto.data, auto.pattern


def _adaboost_core(train_file: str) -> Tuple[TabularData, pattern_type]:
    config: Dict[str, Any] = {"data_config": {"label_name": "Survived"}}
    if CI:
        config.update({"min_epoch": 1, "num_epoch": 2, "max_epoch": 4})
    ensemble = cflearn.Ensemble("clf", config)
    results = ensemble.adaboost(train_file, model=model)
    return results.data, results.pattern


def _test(name: str, _core: Callable[[str], Tuple[TabularData, pattern_type]]) -> None:
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


def test_hpo() -> None:
    _test("hpo", _hpo_core)
    _test("optuna", _optuna_core)


def test_adaboost() -> None:
    _test("adaboost", _adaboost_core)


if __name__ == "__main__":
    test_adaboost()
    test_hpo()
