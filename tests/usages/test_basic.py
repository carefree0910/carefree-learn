import os
import cflearn

import numpy as np

from cfdata.tabular import *


file_folder = os.path.dirname(__file__)
data_folder = os.path.abspath(os.path.join(file_folder, "sick"))
tr_file, cv_file, te_file = map(
    os.path.join, 3 * [data_folder], ["train.txt", "valid.txt", "test.txt"]
)

kwargs = {"min_epoch": 1, "num_epoch": 2, "max_epoch": 4}


def test_array_dataset():
    model = "fcnn"
    dataset = TabularDataset.iris()
    m = cflearn.make(
        model,
        cv_split=0.0,
        use_amp=True,
        data_config={"numerical_columns": list(range(dataset.x.shape[1]))},
        optimizer="adam",
        optimizer_config={"lr": 3e-4},
        metrics=["acc", "auc"],
        model_config={
            "hidden_units": [100],
            "default_encoding_method": "one_hot",
            "mapping_configs": {"batch_norm": True, "dropout": 0.0},
        },
        **kwargs,
    )
    m.fit(*dataset.xy, sample_weights=np.random.random(len(dataset.x)))
    cflearn.estimate(*dataset.xy, wrappers=m)
    cflearn.save(m)
    cflearn.estimate(*dataset.xy, wrappers=cflearn.load())
    cflearn._remove()


def test_file_dataset():
    fcnn = cflearn.make(**kwargs)
    tree_dnn = cflearn.make("tree_dnn", **kwargs)
    fcnn.fit(tr_file, x_cv=cv_file)
    tree_dnn.fit(tr_file, x_cv=cv_file)
    wrappers = [fcnn, tree_dnn]
    cflearn.estimate(tr_file, wrappers=wrappers, contains_labels=True)
    cflearn.estimate(cv_file, wrappers=wrappers, contains_labels=True)
    cflearn.estimate(te_file, wrappers=wrappers, contains_labels=True)
    cflearn.save(wrappers)
    wrappers = cflearn.load()
    cflearn.estimate(tr_file, wrappers=wrappers, contains_labels=True)
    cflearn.estimate(cv_file, wrappers=wrappers, contains_labels=True)
    cflearn.estimate(te_file, wrappers=wrappers, contains_labels=True)
    cflearn._remove()

    # Distributed
    num_repeat = 3
    models = ["nnb", "ndt"]
    for num_jobs in [0, 2]:
        results = cflearn.repeat_with(
            tr_file,
            x_cv=cv_file,
            models=models,
            num_repeat=num_repeat,
            num_jobs=num_jobs,
            temp_folder="__test_file_dataset__",
            **kwargs,
        )
    (tr_x, tr_y), (cv_x, cv_y), (te_x, te_y) = map(
        results.data.read_file, [tr_file, cv_file, te_file]
    )
    tr_y, cv_y, te_y = map(results.data.transform_labels, [tr_y, cv_y, te_y])
    ensembles = {
        model: cflearn.ensemble(model_list)
        for model, model_list in results.patterns.items()
    }
    other_patterns = {}
    for model in models:
        repeat_key = f"{model}_{num_repeat}"
        other_patterns[repeat_key] = results.patterns[model]
        other_patterns[f"{repeat_key}_ensemble"] = ensembles[model]
    cflearn.estimate(tr_x, tr_y, wrappers=wrappers, other_patterns=other_patterns)
    cflearn.estimate(cv_x, cv_y, wrappers=wrappers, other_patterns=other_patterns)
    cflearn.estimate(te_x, te_y, wrappers=wrappers, other_patterns=other_patterns)


if __name__ == "__main__":
    test_array_dataset()
    test_file_dataset()
