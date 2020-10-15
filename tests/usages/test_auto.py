import cflearn

from cfdata.tabular import *

iris = TabularDataset.iris()
iris = TabularData.from_dataset(iris)
split = iris.split(0.1)
train, valid = split.remained, split.split
x_tr, y_tr = train.processed.xy
x_cv, y_cv = valid.processed.xy
data = x_tr, y_tr, x_cv, y_cv


def test_auto():
    fcnn = cflearn.make(use_tqdm=False).fit(*data)

    auto = cflearn.Auto(TaskTypes.CLASSIFICATION).fit(*data)
    predictions = auto.predict(x_cv)
    print("accuracy:", (y_cv == predictions).mean())

    cflearn.estimate(
        x_cv,
        y_cv,
        wrappers=fcnn,
        other_patterns={"auto": auto.pattern},
    )

    export_folder = "iris_vis"
    auto.plot_param_importances(export_folder=export_folder)
    auto.plot_intermediate_values(export_folder=export_folder)


if __name__ == "__main__":
    test_auto()
