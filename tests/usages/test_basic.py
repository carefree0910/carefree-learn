import os
import cflearn

from cfdata.tabular import *


file_folder = os.path.dirname(__file__)
data_folder = os.path.abspath(os.path.join(file_folder, os.pardir, "sick"))
tr_file, cv_file, te_file = map(os.path.join, 3 * [data_folder], ["train.txt", "valid.txt", "test.txt"])


def test_array_dataset():
    model = "fcnn"
    dataset = TabularDataset.iris()
    m = cflearn.make(
        model,
        cv_split=0.,
        data_config={"numerical_columns": list(range(dataset.x.shape[1]))},
        min_epoch=100,
        num_epoch=200,
        max_epoch=400,
        optimizer="adam",
        optimizer_config={"lr": 3e-4},
        metrics=["acc", "auc"],
        model_config={
            "hidden_units": [100],
            "default_encoding_method": "one_hot",
            "mapping_configs": {"batch_norm": True, "dropout": 0.}
        }
    )
    m.fit(*dataset.xy)
    cflearn.estimate(*dataset.xy, wrappers=m)
    cflearn.save(m)
    cflearn.estimate(*dataset.xy, wrappers=cflearn.load())
    cflearn._remove()


def test_file_dataset():
    fcnn = cflearn.make()
    tree_dnn = cflearn.make("tree_dnn")
    fcnn.fit(tr_file, x_cv=cv_file)
    tree_dnn.fit(tr_file, x_cv=cv_file)
    wrappers = [fcnn, tree_dnn]
    cflearn.estimate(tr_file, wrappers=wrappers)
    cflearn.estimate(cv_file, wrappers=wrappers)
    cflearn.estimate(te_file, wrappers=wrappers)
    cflearn.save(wrappers)
    wrappers = cflearn.load()
    cflearn.estimate(tr_file, wrappers=wrappers)
    cflearn.estimate(cv_file, wrappers=wrappers)
    cflearn.estimate(te_file, wrappers=wrappers)
    cflearn._remove()

    # Distributed
    num_repeat = 3
    models = ["nnb", "ndt"]
    for num_jobs in [0, 2]:
        transformer, results = cflearn.repeat_with(
            tr_file, x_cv=cv_file, models=models,
            num_repeat=num_repeat, num_jobs=num_jobs
        )
    (tr_x, tr_y), (cv_x, cv_y), (te_x, te_y) = map(transformer.get_xy, [tr_file, cv_file, te_file])
    ensembles = {model: cflearn.ensemble(model_list) for model, model_list in results.items()}
    other_patterns = {}
    for model in models:
        repeat_key = f"{model}_{num_repeat}"
        other_patterns[repeat_key] = results[model]
        other_patterns[f"{repeat_key}_ensemble"] = ensembles[model]
    cflearn.estimate(tr_x, tr_y, wrappers=wrappers, other_patterns=other_patterns)
    cflearn.estimate(cv_x, cv_y, wrappers=wrappers, other_patterns=other_patterns)
    cflearn.estimate(te_x, te_y, wrappers=wrappers, other_patterns=other_patterns)


def test_hpo():
    for num_parallel in [0, 2]:
        cflearn.tune_with(
            tr_file,
            x_cv=cv_file,
            task_type=TaskTypes.CLASSIFICATION,
            num_repeat=2, num_parallel=num_parallel, num_search=10
        )


if __name__ == '__main__':
    test_array_dataset()
    test_file_dataset()
    test_hpo()
