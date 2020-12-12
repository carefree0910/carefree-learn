import os
import pickle
import cflearn
import numpy as np

if __name__ == "__main__":
    file_folder = os.path.dirname(__file__)
    iris_data_file = os.path.join(file_folder, "iris.data")
    sklearn_runner_file = os.path.join(file_folder, "run_sklearn.py")

    m = cflearn.make().fit(iris_data_file)
    print(m.tr_data.raw.x[0])
    print(m.tr_data.raw.y[0])
    print(m.tr_data.processed.x[0])
    print(m.tr_data.processed.y[0])

    tr_x = m.tr_data.processed.x
    cv_x = m.cv_data.processed.x
    stacked = np.vstack([tr_x, cv_x])
    print(stacked.mean(0))
    print(stacked.std(0))

    predictions = m.predict(iris_data_file, contains_labels=True)
    cflearn.evaluate(iris_data_file, pipelines=m)

    result = cflearn.repeat_with(iris_data_file, num_repeat=3)
    cflearn.evaluate(iris_data_file, pipelines=result.pipelines)
    cflearn._rmtree("__tmp__")

    models = ["linear", "fcnn"]
    result = cflearn.repeat_with(iris_data_file, models=models, num_repeat=3)
    cflearn.evaluate(iris_data_file, pipelines=result.pipelines)
    cflearn._rmtree("__tmp__")

    result = cflearn.repeat_with(iris_data_file, num_repeat=10, num_jobs=2)

    experiment = cflearn.Experiment()
    tr_x, tr_y = m.tr_data.processed.xy
    cv_x, cv_y = m.cv_data.processed.xy
    data_folder = experiment.dump_data_bundle(tr_x, tr_y, cv_x, cv_y)

    for model in ["linear", "fcnn"]:
        experiment.add_task(model=model, data_folder=data_folder)
    run_command = f"python {sklearn_runner_file}"
    common_kwargs = {"run_command": run_command, "data_folder": data_folder}
    experiment.add_task(model="decision_tree", **common_kwargs)
    experiment.add_task(model="random_forest", **common_kwargs)

    results = experiment.run_tasks()

    pipelines = {}
    sk_patterns = {}
    for workplace, workplace_key in zip(results.workplaces, results.workplace_keys):
        model = workplace_key[0]
        if model not in ["decision_tree", "random_forest"]:
            pipelines[model] = cflearn.task_loader(workplace)
        else:
            model_file = os.path.join(workplace, "sk_model.pkl")
            with open(model_file, "rb") as f:
                sk_model = pickle.load(f)
                # In `carefree-learn`, we treat labels as column vectors.
                # So we need to reshape the outputs from the scikit-learn models.
                sk_predict = lambda x: sk_model.predict(x).reshape([-1, 1])
                sk_predict_prob = lambda x: sk_model.predict_proba(x)
                sk_pattern = cflearn.ModelPattern(
                    predict_method=sk_predict,
                    predict_prob_method=sk_predict_prob,
                )
                sk_patterns[model] = sk_pattern

    cflearn.evaluate(cv_x, cv_y, pipelines=pipelines, other_patterns=sk_patterns)

    auto = cflearn.Auto("clf", models="fcnn")
    auto.fit(tr_x, tr_y, cv_x, cv_y)

    predictions = auto.predict(cv_x)
    print("accuracy:", (predictions == cv_y).mean())

    all_patterns = sk_patterns.copy()
    all_patterns["auto"] = auto.pattern
    cflearn.evaluate(cv_x, cv_y, pipelines=pipelines, other_patterns=all_patterns)

    auto.pack("pack")

    unpacked = cflearn.Auto.unpack("pack")
    predictions = unpacked.pattern.predict(cv_x)
