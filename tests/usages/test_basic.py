import os
import cflearn

import numpy as np

from cfdata.tabular import TaskTypes
from cfdata.tabular import TabularData
from cfdata.tabular import TabularDataset


file_folder = os.path.dirname(__file__)
data_folder = os.path.abspath(os.path.join(file_folder, "sick"))
tr_file, cv_file, te_file = map(
    os.path.join,
    3 * [data_folder],
    ["train.txt", "valid.txt", "test.txt"],
)

kwargs = {"min_epoch": 1, "num_epoch": 2, "max_epoch": 4}


def test_array_dataset() -> None:
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
        **kwargs,  # type: ignore
    )
    m.fit(*dataset.xy, sample_weights=np.random.random(len(dataset.x)))
    cflearn.estimate(*dataset.xy, pipelines=m)
    cflearn.save(m)
    cflearn.estimate(*dataset.xy, pipelines=cflearn.load())
    cflearn._remove()


def test_file_dataset() -> None:
    fcnn = cflearn.make(**kwargs)  # type: ignore
    tree_dnn = cflearn.make("tree_dnn", **kwargs)  # type: ignore
    fcnn.fit(tr_file, x_cv=cv_file)
    tree_dnn.fit(tr_file, x_cv=cv_file)
    pipeline_list = [fcnn, tree_dnn]
    cflearn.estimate(tr_file, pipelines=pipeline_list, contains_labels=True)
    cflearn.estimate(cv_file, pipelines=pipeline_list, contains_labels=True)
    cflearn.estimate(te_file, pipelines=pipeline_list, contains_labels=True)
    cflearn.save(pipeline_list)
    pipelines_dict = cflearn.load()
    cflearn.estimate(tr_file, pipelines=pipelines_dict, contains_labels=True)
    cflearn.estimate(cv_file, pipelines=pipelines_dict, contains_labels=True)
    cflearn.estimate(te_file, pipelines=pipelines_dict, contains_labels=True)
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
            **kwargs,  # type: ignore
        )
    data = results.data
    patterns = results.patterns
    assert isinstance(data, TabularData)
    assert patterns is not None
    (tr_x, tr_y), (cv_x, cv_y), (te_x, te_y) = map(
        data.read_file,
        [tr_file, cv_file, te_file],
    )
    tr_y, cv_y, te_y = map(data.transform_labels, [tr_y, cv_y, te_y])
    ensembles = {
        model: cflearn.ensemble(model_list) for model, model_list in patterns.items()
    }
    other_patterns = {}
    for model in models:
        repeat_key = f"{model}_{num_repeat}"
        other_patterns[repeat_key] = patterns[model]
        other_patterns[f"{repeat_key}_ensemble"] = ensembles[model]
    cflearn.estimate(
        tr_x,
        tr_y,
        pipelines=pipelines_dict,
        other_patterns=other_patterns,
    )
    cflearn.estimate(
        cv_x,
        cv_y,
        pipelines=pipelines_dict,
        other_patterns=other_patterns,
    )
    cflearn.estimate(
        te_x,
        te_y,
        pipelines=pipelines_dict,
        other_patterns=other_patterns,
    )


def test_auto_file() -> None:
    # TODO : in ONNX, device may be mixed up because Pruner's mask will be on cuda:0
    kwargs = {"min_epoch": 1, "num_epoch": 2, "max_epoch": 4}
    auto = cflearn.Auto(TaskTypes.CLASSIFICATION)
    predict_config = {"contains_labels": True}
    auto.fit(
        tr_file,
        x_cv=cv_file,
        num_trial=4,
        num_repeat=1,
        predict_config=predict_config,
        extra_config=kwargs.copy(),
    )
    pred1 = auto.predict(te_file)
    prob1 = auto.predict_prob(te_file)
    export_folder = "packed"
    auto.pack(export_folder)
    predictors, weights = auto.get_predictors(export_folder)
    pattern = cflearn.Pack.ensemble(predictors, weights, **predict_config)
    pred2 = pattern.predict(te_file)
    prob2 = pattern.predict(te_file, requires_prob=True)
    assert np.allclose(pred1, pred2)
    assert np.allclose(prob1, prob2)


if __name__ == "__main__":
    test_array_dataset()
    test_file_dataset()
