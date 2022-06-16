# type: ignore

import os
import pickle
import cflearn

import numpy as np

from cftool.misc import _rmtree
from cflearn.misc.toolkit import check_is_ci
from cflearn.misc.toolkit import inject_debug


is_ci = check_is_ci()
file_folder = os.path.dirname(__file__)
iris_data_file = os.path.join(file_folder, "iris.data")
sklearn_runner_file = os.path.join(file_folder, "run_sklearn.py")


if __name__ == "__main__":
    metrics = ["acc", "auc"]
    m = cflearn.api.fit_ml(iris_data_file, carefree=True, debug=is_ci)
    print(m.cf_data.raw.x[0])
    print(m.cf_data.raw.y[0])
    print(m.cf_data.processed.x[0])
    print(m.cf_data.processed.y[0])

    data = m.data
    x_train, y_train = data.train_cf_data.processed.xy
    x_valid, y_valid = data.valid_cf_data.processed.xy
    stacked = np.vstack([x_train, x_valid])
    print(stacked.mean(0))
    print(stacked.std(0))

    idata = cflearn.MLInferenceData(iris_data_file)
    predictions = m.predict(idata, contains_labels=True)
    cflearn.ml.evaluate(idata, metrics=metrics, pipelines=m)

    kwargs = dict(carefree=True, num_repeat=2)
    result = cflearn.api.repeat_ml(iris_data_file, debug=is_ci, **kwargs)
    cflearn.ml.evaluate(data, metrics=metrics, pipelines=result.pipelines)

    models = ["linear", "fcnn"]
    kwargs = dict(carefree=True, models=models, num_repeat=2, num_jobs=2)
    result = cflearn.api.repeat_ml(iris_data_file, debug=is_ci, **kwargs)
    cflearn.ml.evaluate(data, metrics=metrics, pipelines=result.pipelines)

    experiment = cflearn.dist.ml.Experiment()
    data_folder = experiment.dump_data_bundle(x_train, y_train, x_valid, y_valid)

    config = {}
    if is_ci:
        inject_debug(config)
    experiment.add_task(model="fcnn", config=config, data_folder=data_folder)
    experiment.add_task(model="linear", config=config, data_folder=data_folder)
    run_command = f"python {sklearn_runner_file}"
    common_kwargs = {"run_command": run_command, "data_folder": data_folder}
    experiment.add_task(model="decision_tree", **common_kwargs)  # type: ignore
    experiment.add_task(model="random_forest", **common_kwargs)  # type: ignore

    results = experiment.run_tasks()

    pipelines = {}
    sk_patterns = {}
    for workplace, workplace_key in zip(results.workplaces, results.workplace_keys):
        model = workplace_key[0]
        if model not in ["decision_tree", "random_forest"]:
            pipelines[model] = cflearn.ml.task_loader(
                workplace,
                cflearn.ml.CarefreePipeline,
            )
        else:
            model_file = os.path.join(workplace, "sk_model.pkl")
            with open(model_file, "rb") as f:
                sk_model = pickle.load(f)
                # In `carefree-learn`, we treat labels as column vectors.
                # So we need to reshape the outputs from the scikit-learn models.
                sk_predict = lambda d: sk_model.predict(d.x_train).reshape([-1, 1])
                sk_predict_prob = lambda d: sk_model.predict_proba(d.x_train)
                sk_pattern = cflearn.ml.ModelPattern(
                    predict_method=sk_predict,
                    predict_prob_method=sk_predict_prob,
                )
                sk_patterns[model] = sk_pattern

    cflearn.ml.evaluate(
        cflearn.MLInferenceData(x_valid, y_valid),
        metrics=metrics,
        pipelines=pipelines,
        other_patterns=sk_patterns,
    )

    _rmtree("_logs")
    _rmtree("_repeat")
    _rmtree("_parallel_")
    _rmtree("__experiment__")
