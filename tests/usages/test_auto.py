import os
import cflearn
import platform

from cftool.misc import shallow_copy_dict
from cfdata.tabular import TabularData
from cfdata.tabular import TabularDataset

IS_LINUX = platform.system() == "Linux"

iris = TabularDataset.iris()
iris = TabularData.from_dataset(iris)
split = iris.split(0.1)
train, valid = split.remained, split.split
x_tr, y_tr = train.processed.xy
x_cv, y_cv = valid.processed.xy
data = x_tr, y_tr, x_cv, y_cv

CI = True
model = "fcnn"
logging_folder = "__test_auto__"
num_jobs_list = [0] if IS_LINUX else [0, 1, 2]
kwargs = {"min_epoch": 1, "num_epoch": 2, "max_epoch": 4} if CI else {}
kwargs["logging_folder"] = logging_folder


def test_auto() -> None:
    for num_jobs in num_jobs_list:
        local_kwargs = shallow_copy_dict(kwargs)
        local_logging_folder = os.path.join(logging_folder, str(num_jobs))
        local_kwargs["logging_folder"] = local_logging_folder
        fcnn = cflearn.make(use_tqdm=False, **local_kwargs).fit(*data)  # type: ignore

        auto = cflearn.Auto("clf", models=model)
        auto.fit(
            *data,
            num_trial=10,
            num_jobs=num_jobs,
            num_final_repeat=3,
            temp_folder=logging_folder,
            extra_config=shallow_copy_dict(local_kwargs),
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
        auto.plot_param_importances(model, export_folder=export_folder)
        auto.plot_intermediate_values(model, export_folder=export_folder)

        cflearn._rmtree(local_logging_folder)
        cflearn._rmtree(export_folder)


if __name__ == "__main__":
    test_auto()
