import cflearn

from cfdata.tabular import TaskTypes
from cfdata.tabular import TabularData
from cfdata.tabular import TabularDataset

iris = TabularDataset.iris()
iris = TabularData.from_dataset(iris)
split = iris.split(0.1)
train, valid = split.remained, split.split
x_tr, y_tr = train.processed.xy
x_cv, y_cv = valid.processed.xy
data = x_tr, y_tr, x_cv, y_cv

CI = True
kwargs = {"min_epoch": 1, "num_epoch": 2, "max_epoch": 4} if CI else {}


def test_auto() -> None:
    for num_jobs in [1, 2]:
        fcnn = cflearn.make(use_tqdm=False, **kwargs).fit(*data)  # type: ignore

        auto = cflearn.Auto(TaskTypes.CLASSIFICATION)
        auto.fit(
            *data,
            num_jobs=num_jobs,
            temp_folder=f"__test_auto_{num_jobs}__",
            extra_config=kwargs.copy(),
        )
        predictions = auto.predict(x_cv)
        print("accuracy:", (y_cv == predictions).mean())

        cflearn.estimate(
            x_cv,
            y_cv,
            pipelines=fcnn,
            other_patterns={"auto": auto.pattern},
        )

        export_folder = "iris_vis"
        auto.plot_param_importances(export_folder=export_folder)
        auto.plot_intermediate_values(export_folder=export_folder)


if __name__ == "__main__":
    test_auto()
