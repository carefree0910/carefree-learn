import os
import cflearn

from cfdata.tabular import *

file_folder = os.path.dirname(__file__)


def test():
    train_file = os.path.join(file_folder, "train.csv")
    test_file = os.path.join(file_folder, "test.csv")
    data_config = {"label_name": "Survived"}
    hpo = cflearn.tune_with(
        train_file,
        model="tree_dnn",
        task_type=TaskTypes.CLASSIFICATION,
        data_config=data_config,
        num_parallel=0
    )
    results = cflearn.repeat_with(
        train_file,
        **hpo.best_param,
        models="tree_dnn",
        num_repeat=10, num_jobs=0,
        data_config=data_config
    )
    ensemble = cflearn.EnsemblePattern(results.patterns["tree_dnn"])
    predictions = ensemble.predict(test_file).ravel()
    x_te, _ = results.transformer.data.read_file(test_file, contains_labels=False)
    id_list = DataTuple.with_transpose(x_te, None).xT[0]
    # Score : achieved ~0.79
    with open("submissions.csv", "w") as f:
        f.write("PassengerId,Survived\n")
        for test_id, prediction in zip(id_list, predictions):
            f.write(f"{test_id},{prediction}\n")


if __name__ == '__main__':
    test()
