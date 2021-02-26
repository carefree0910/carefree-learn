import os
import cflearn

import numpy as np

from cftool.misc import fix_float_to_length
from cfdata.tabular import TabularData
from cfdata.tabular import TabularDataset


file_folder = os.path.dirname(__file__)
data_folder = os.path.abspath(os.path.join(file_folder, "sick"))
tr_file, cv_file, te_file = map(
    os.path.join,
    3 * [data_folder],
    ["train.txt", "valid.txt", "test.txt"],
)

logging_folder = "__test_basic__"
# TODO : Fix `ndt` when `num_jobs == 2`, `ndt` is not deterministic now
num_jobs_list = [0]
kwargs = {
    "min_epoch": 1,
    "num_epoch": 2,
    "max_epoch": 4,
    "logging_folder": logging_folder,
}


def test_array_dataset() -> None:
    model = "fcnn"
    # fake
    n = 10000
    x = np.random.random([n, 420])
    y = np.random.random([n, 1])
    for use_simplify_data in [True, False]:
        task_type = "reg" if use_simplify_data else ""
        m = cflearn.make(
            model,
            task_type=task_type,
            use_simplify_data=use_simplify_data,
            **kwargs,  # type: ignore
        )
        m.fit(x, y)
        cflearn._rmtree(logging_folder)
    # iris
    dataset = TabularDataset.iris()
    m = cflearn.make(
        model,
        cv_split=0.0,
        use_amp=True,
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
    cflearn.evaluate(*dataset.xy, pipelines=m)
    cflearn.save(m)
    m2 = cflearn.load()[model][0]
    assert m.tr_data == m2.tr_data
    assert m.cv_data == m2.cv_data
    cflearn.evaluate(*dataset.xy, pipelines=m2)
    cflearn._rmtree(logging_folder)
    cflearn._remove()
    data = TabularData.from_dataset(dataset)
    split = data.split(0.1)
    x_tr, y_tr = split.remained.processed.xy
    x_cv, y_cv = split.split.processed.xy
    sample_weights = np.random.random(len(dataset))
    m = cflearn.make(model, use_tqdm=False, **kwargs)  # type: ignore
    m.fit(x_tr, y_tr, x_cv, y_cv, sample_weights=sample_weights)
    cflearn.save(m)
    m2 = cflearn.load()[model][0]
    assert m.tr_data == m2.tr_data
    assert m.cv_data == m2.cv_data
    assert np.allclose(sample_weights, m2.sample_weights)
    cflearn._rmtree(logging_folder)
    cflearn._remove()


def test_file_dataset() -> None:
    fcnn = cflearn.make(**kwargs)  # type: ignore
    tree_dnn = cflearn.make("tree_dnn", **kwargs)  # type: ignore
    fcnn.fit(tr_file, x_cv=cv_file)
    tree_dnn.fit(tr_file, x_cv=cv_file)
    pipeline_list = [fcnn, tree_dnn]
    cflearn.evaluate(tr_file, pipelines=pipeline_list, contains_labels=True)
    cflearn.evaluate(cv_file, pipelines=pipeline_list, contains_labels=True)
    cflearn.evaluate(te_file, pipelines=pipeline_list, contains_labels=True)
    cflearn.save(pipeline_list)
    pipelines_dict = cflearn.load()
    cflearn.evaluate(tr_file, pipelines=pipelines_dict, contains_labels=True)
    cflearn.evaluate(cv_file, pipelines=pipelines_dict, contains_labels=True)
    cflearn.evaluate(te_file, pipelines=pipelines_dict, contains_labels=True)
    cflearn._rmtree(logging_folder)
    cflearn._remove()

    # Distributed
    results = None
    num_repeat = 3
    models = ["nnb", "ndt"]
    repeat_kwargs = kwargs.copy()
    repeat_kwargs.pop("logging_folder")
    for num_jobs in num_jobs_list:
        results = cflearn.repeat_with(
            tr_file,
            x_cv=cv_file,
            models=models,
            num_repeat=num_repeat,
            num_jobs=num_jobs,
            temp_folder=os.path.join(logging_folder, str(num_jobs)),
            **repeat_kwargs,  # type: ignore
        )
        cflearn._rmtree(logging_folder)

    assert results is not None
    data = results.data
    patterns = results.patterns
    assert isinstance(data, TabularData)
    assert patterns is not None
    (tr_x_df, tr_y_df), (cv_x_df, cv_y_df), (te_x_df, te_y_df) = map(
        data.read_file,
        [tr_file, cv_file, te_file],
    )
    tr_x = tr_x_df.to_numpy()
    tr_y = tr_y_df.to_numpy()
    cv_x = cv_x_df.to_numpy()
    cv_y = cv_y_df.to_numpy()
    te_x = te_x_df.to_numpy()
    te_y = te_y_df.to_numpy()
    tr_y, cv_y, te_y = map(data.transform_labels, [tr_y, cv_y, te_y])
    ensembles = {
        model: cflearn.Ensemble.stacking(model_list)
        for model, model_list in patterns.items()
    }
    other_patterns = {}
    for model in models:
        repeat_key = f"{model}_{num_repeat}"
        other_patterns[repeat_key] = patterns[model]
        other_patterns[f"{repeat_key}_ensemble"] = ensembles[model]
    cflearn.evaluate(
        tr_x,
        tr_y,
        pipelines=pipelines_dict,
        other_patterns=other_patterns,
    )
    cflearn.evaluate(
        cv_x,
        cv_y,
        pipelines=pipelines_dict,
        other_patterns=other_patterns,
    )
    cflearn.evaluate(
        te_x,
        te_y,
        pipelines=pipelines_dict,
        other_patterns=other_patterns,
    )


def test_file_dataset2() -> None:
    config = {
        "optimizer": "adam",
        "optimizer_config": {"lr": 0.00046240517013441517},
        "model_config": {
            "ema_decay": 0.9,
            "default_encoding_configs": {
                "init_method": "truncated_normal",
                "embedding_dim": 8,
            },
            "pipe_configs": {
                "fcnn": {
                    "head": {
                        "mapping_configs": {
                            "batch_norm": True,
                            "dropout": 0.12810052348936224,
                            "pruner_config": {"method": "surgery"},
                        },
                    }
                }
            },
        },
    }
    config.update(kwargs)  # type: ignore

    m = cflearn.make(**config).fit(tr_file, x_cv=cv_file)  # type: ignore
    save_name = "__file_trained__"
    cflearn.save(m, save_name)
    m2 = cflearn.load(save_name)["fcnn"][0]
    pack_name = "__file_packed__"
    cflearn.Pack.pack(m2, pack_name)
    m3 = cflearn.Pack.get_predictor(pack_name)

    assert m.binary_threshold is not None
    b1 = fix_float_to_length(m.binary_threshold, 6)
    b2 = fix_float_to_length(m2.binary_threshold, 6)
    b3 = fix_float_to_length(m3.inference.binary_threshold, 6)
    assert b1 == b2 == b3

    pred1 = m.predict(te_file, contains_labels=True)
    pred2 = m2.predict(te_file, contains_labels=True)
    pred3 = m3.predict(te_file, contains_labels=True)
    pred4 = m3.to_pattern(contains_labels=True).predict(te_file)
    assert np.allclose(pred1, pred2)
    assert np.allclose(pred2, pred3)
    assert np.allclose(pred3, pred4)

    prob1 = m.predict_prob(te_file, contains_labels=True)
    prob2 = m2.predict_prob(te_file, contains_labels=True)
    prob3 = m3.predict_prob(te_file, contains_labels=True)
    prob4 = m3.to_pattern(contains_labels=True).predict(te_file, requires_prob=True)
    assert np.allclose(prob1, prob2, atol=1e-4, rtol=1e-4)
    assert np.allclose(prob2, prob3, atol=1e-4, rtol=1e-4)
    assert np.allclose(prob3, prob4, atol=1e-4, rtol=1e-4)

    cflearn._remove(save_name)
    cflearn._rmtree(logging_folder)
    os.remove(f"{pack_name}.zip")


def test_auto_file() -> None:
    models = ["linear", "fcnn", "tree_dnn", "tree_linear", "nnb", "ndt"]
    auto = cflearn.Auto("clf", models=models)
    predict_config = {"contains_labels": True}
    extra_config = kwargs.copy()
    extra_config.pop("logging_folder")
    auto.fit(
        tr_file,
        x_cv=cv_file,
        num_trial=3,
        num_repeat=1,
        num_final_repeat=3,
        temp_folder=logging_folder,
        predict_config=predict_config,
        extra_config=extra_config,
    )
    pred1 = auto.predict(te_file)
    prob1 = auto.predict_prob(te_file)
    export_name = "packed"
    auto.pack(export_name)
    unpacked = cflearn.Auto.unpack(export_name, **predict_config)
    pattern = unpacked.pattern
    pred2 = pattern.predict(te_file)
    prob2 = pattern.predict(te_file, requires_prob=True)
    assert np.allclose(pred1, pred2, equal_nan=True)
    assert np.allclose(prob1, prob2, equal_nan=True, atol=1e-2, rtol=1e-2)
    cflearn._rmtree(logging_folder)
    os.remove(f"{export_name}.zip")


if __name__ == "__main__":
    test_array_dataset()
    test_file_dataset()
    test_file_dataset2()
    test_auto_file()
