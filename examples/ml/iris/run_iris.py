import os
import pickle
import cflearn

import numpy as np

from cflearn.misc.toolkit import _rmtree


file_folder = os.path.dirname(__file__)
iris_data_file = os.path.join(file_folder, "iris.data")
sklearn_runner_file = os.path.join(file_folder, "run_sklearn.py")


if __name__ == "__main__":
    metrics = ["acc", "auc"]
    m = cflearn.ml.CarefreePipeline()
    m.fit(iris_data_file)
    print(m.data.raw.x[0])
    print(m.data.raw.y[0])
    print(m.data.processed.x[0])
    print(m.data.processed.y[0])

    train_x, train_y = m.train_data.processed.xy
    valid_x, valid_y = m.valid_data.processed.xy  # type: ignore
    stacked = np.vstack([train_x, valid_x])
    print(stacked.mean(0))
    print(stacked.std(0))

    predictions = m.predict(iris_data_file, contains_labels=True)
    cflearn.ml.evaluate(iris_data_file, metrics=metrics, pipelines=m)

    result = cflearn.ml.repeat_with(
        iris_data_file,
        pipeline_base=cflearn.ml.CarefreePipeline,
        num_repeat=2,
    )
    cflearn.ml.evaluate(iris_data_file, metrics=metrics, pipelines=result.pipelines)

    models = ["linear", "fcnn"]
    result = cflearn.ml.repeat_with(
        iris_data_file,
        pipeline_base=cflearn.ml.CarefreePipeline,
        models=models,
        num_repeat=2,
        num_jobs=2,
    )
    cflearn.ml.evaluate(iris_data_file, metrics=metrics, pipelines=result.pipelines)

    experiment = cflearn.dist.ml.Experiment()
    data_folder = experiment.dump_data_bundle(train_x, train_y, valid_x, valid_y)

    experiment.add_task(model="fcnn", data_folder=data_folder)
    experiment.add_task(model="linear", data_folder=data_folder)
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
                sk_predict = lambda x: sk_model.predict(x).reshape([-1, 1])
                sk_predict_prob = lambda x: sk_model.predict_proba(x)
                sk_pattern = cflearn.ml.ModelPattern(
                    predict_method=sk_predict,
                    predict_prob_method=sk_predict_prob,
                )
                sk_patterns[model] = sk_pattern

    cflearn.ml.evaluate(
        valid_x,
        valid_y,
        metrics=metrics,
        pipelines=pipelines,
        other_patterns=sk_patterns,
    )

    _rmtree("_logs")
    _rmtree("_repeat")
    _rmtree("_parallel_")
    _rmtree("__experiment__")
