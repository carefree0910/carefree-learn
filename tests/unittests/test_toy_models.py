import cflearn
import unittest

x_numerical = [[1.2], [3.4], [5.6]]
x_categorical = [[1.0], [3.0], [5.0]]
x_mix = [xn + xc for xn, xc in zip(x_numerical, x_categorical)]
y_clf = [[1], [1], [2]]
y_reg = [[1.3], [3.5], [5.7]]


class TestToy(unittest.TestCase):
    def test_linear_toy(self) -> None:
        cflearn.make_toy_model("linear", task_type="clf", data_tuple=(x_mix, y_clf))
        cflearn.make_toy_model(
            "linear", task_type="clf", data_tuple=(x_numerical, y_clf)
        )
        cflearn.make_toy_model(
            "linear", task_type="clf", data_tuple=(x_categorical, y_clf)
        )
        cflearn.make_toy_model("linear", data_tuple=(x_mix, y_reg))
        cflearn.make_toy_model("linear", data_tuple=(x_numerical, y_reg))
        cflearn.make_toy_model("linear", data_tuple=(x_categorical, y_reg))
        cflearn._rmtree("_logging")

    def test_fcnn_toy(self) -> None:
        cflearn.make_toy_model(task_type="clf", data_tuple=(x_mix, y_clf))
        cflearn.make_toy_model(task_type="clf", data_tuple=(x_numerical, y_clf))
        cflearn.make_toy_model(task_type="clf", data_tuple=(x_categorical, y_clf))
        cflearn.make_toy_model(data_tuple=(x_mix, y_reg))
        cflearn.make_toy_model(data_tuple=(x_numerical, y_reg))
        cflearn.make_toy_model(data_tuple=(x_categorical, y_reg))
        cflearn._rmtree("_logging")

    def test_nnb_toy(self) -> None:
        cflearn.make_toy_model("nnb", task_type="clf", data_tuple=(x_mix, y_clf))
        cflearn.make_toy_model("nnb", task_type="clf", data_tuple=(x_numerical, y_clf))
        cflearn.make_toy_model(
            "nnb", task_type="clf", data_tuple=(x_categorical, y_clf)
        )
        cflearn._rmtree("_logging")

    def test_ndt_toy(self) -> None:
        cflearn.make_toy_model("ndt", task_type="clf", data_tuple=(x_mix, y_clf))
        cflearn.make_toy_model("ndt", task_type="clf", data_tuple=(x_numerical, y_clf))
        cflearn.make_toy_model(
            "ndt", task_type="clf", data_tuple=(x_categorical, y_clf)
        )
        cflearn._rmtree("_logging")

    def test_tree_dnn_toy(self) -> None:
        cflearn.make_toy_model("tree_dnn", task_type="clf", data_tuple=(x_mix, y_clf))
        cflearn.make_toy_model(
            "tree_dnn", task_type="clf", data_tuple=(x_numerical, y_clf)
        )
        cflearn.make_toy_model(
            "tree_dnn", task_type="clf", data_tuple=(x_categorical, y_clf)
        )
        cflearn.make_toy_model("tree_dnn", data_tuple=(x_mix, y_reg))
        cflearn.make_toy_model("tree_dnn", data_tuple=(x_numerical, y_reg))
        cflearn.make_toy_model("tree_dnn", data_tuple=(x_categorical, y_reg))
        cflearn._rmtree("_logging")

    def test_ddr_toy(self) -> None:
        cflearn.make_toy_model("ddr", data_tuple=(x_mix, y_reg))
        cflearn.make_toy_model("ddr", data_tuple=(x_numerical, y_reg))
        cflearn.make_toy_model("ddr", data_tuple=(x_categorical, y_reg))
        cfg = {"model_config": {"feature_units": []}}
        cflearn.make_toy_model("ddr", config=cfg, data_tuple=(x_mix, y_reg))
        cflearn.make_toy_model("ddr", config=cfg, data_tuple=(x_numerical, y_reg))
        cflearn.make_toy_model("ddr", config=cfg, data_tuple=(x_categorical, y_reg))
        cflearn._rmtree("_logging")


if __name__ == "__main__":
    unittest.main()
