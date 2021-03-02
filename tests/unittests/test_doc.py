import os
import sys
import torch
import pickle
import cflearn
import platform
import unittest

import numpy as np
import torch.nn as nn

from typing import Any
from typing import Dict
from typing import Optional
from cftool.ml import ModelPattern
from cfdata.tabular import TabularDataset
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from cflearn.types import param_type
from cflearn.types import losses_type
from cflearn.types import tensor_dict_type
from cflearn.misc.toolkit import Activations
from cflearn.misc.toolkit import Initializer
from cflearn.modules.blocks import Linear
from cflearn.modules.blocks import CrossBlock
from cflearn.modules.transform import Dimensions
from cflearn.modules.extractors import Identity

file_folder = os.path.dirname(__file__)
data_folder = os.path.abspath(os.path.join(file_folder, "data"))
examples_folder = os.path.join(file_folder, os.pardir, os.pardir, "examples")
IS_WINDOWS = platform.system() == "Windows"


class TestDoc(unittest.TestCase):
    def test_introduction1(self) -> None:
        toy = cflearn.make_toy_model()
        data = toy.tr_data.converted
        self.assertEqual(data.x.item(), 0.0)
        self.assertEqual(data.y.item(), 1.0)

    def test_introduction2(self) -> None:
        @cflearn.register_processor("plus_one")
        class _(cflearn.Processor):
            @property
            def input_dim(self) -> int:
                return 1

            @property
            def output_dim(self) -> int:
                return 1

            def fit(self, columns: np.ndarray) -> cflearn.Processor:
                return self

            def _process(self, columns: np.ndarray) -> np.ndarray:
                return columns + 1

            def _recover(self, processed_columns: np.ndarray) -> np.ndarray:
                return processed_columns - 1

        config = {"data_config": {"label_process_method": "plus_one"}}
        toy = cflearn.make_toy_model(config=config)
        y = toy.tr_data.converted.y
        processed_y = toy.tr_data.processed.y
        self.assertEqual(y.item(), 1.0)
        self.assertEqual(processed_y.item(), 2.0)

    def test_design_principle(self) -> None:
        @cflearn.register_initializer("all_one")
        def all_one(_: Initializer, parameter: param_type) -> None:
            parameter.fill_(1.0)

        param = nn.Parameter(torch.zeros(3))
        with torch.no_grad():
            Initializer().initialize(param, "all_one")
        self.assertTrue(torch.allclose(param.data, torch.ones_like(param.data)))

    def test_quick_start1(self) -> None:
        x, y = TabularDataset.iris().xy
        m = cflearn.make().fit(x, y)
        m.predict(x)
        m.predict_prob(x)
        cflearn.evaluate(x, y, pipelines=m)
        cflearn.save(m)
        ms = cflearn.load()
        self.assertListEqual(list(ms.keys()), ["fcnn"])
        self.assertEqual(len(list(ms.values())[0]), 1)

    def test_quick_start2(self) -> None:
        m = cflearn.make(delim=",", min_epoch=2000, has_column_names=False)
        xor_file = os.path.join(data_folder, "xor.txt")
        m.fit(xor_file, x_cv=xor_file)
        cflearn.evaluate(xor_file, pipelines=m, contains_labels=True)
        self.assertEqual(m.predict([[0, 0]]).item(), 0)  # type: ignore
        self.assertEqual(m.predict([[0, 1]]).item(), 1)  # type: ignore
        self.assertTrue(
            np.allclose(
                m.predict(xor_file, contains_labels=True),  # type: ignore
                np.array([[0], [1], [1], [0]]),
            )
        )

    def test_quick_start3(self) -> None:
        if not IS_WINDOWS:
            cflearn.make("wnd").draw("wnd.png", transparent=False)

    def test_configurations1(self) -> None:
        config = {"foo": 0, "dummy": 1}
        fcnn = cflearn.make(**config)  # type: ignore
        self.assertEqual(fcnn.foo, 0)
        self.assertEqual(fcnn.dummy, 1)

    def test_configurations2(self) -> None:
        config = os.path.join(data_folder, "basic.json")
        increment_config = {"foo": 2}
        fcnn = cflearn.make(config=config, increment_config=increment_config)
        self.assertEqual(fcnn.foo, 2)
        self.assertEqual(fcnn.dummy, 1)
        fcnn = cflearn.make(config=config)
        self.assertEqual(fcnn.foo, 0)
        self.assertEqual(fcnn.dummy, 1)

    def test_apis1(self) -> None:
        x = np.random.random([1000, 10])
        y = np.random.random([1000, 1])
        m = cflearn.make("linear").fit(x, y)
        skm = LinearSVR().fit(x, y.ravel())
        sk_predict = lambda x_: skm.predict(x_).reshape([-1, 1])
        sk_pattern = ModelPattern(predict_method=sk_predict)
        cflearn.evaluate(x, y, pipelines=m, other_patterns={"sklearn": sk_pattern})

    def test_apis2(self) -> None:
        x = np.random.random([1000, 10])
        y = np.random.random([1000, 1])
        m = cflearn.make().fit(x, y)
        cflearn.save(m)

    def test_apis3(self) -> None:
        x = np.random.random([1000, 10])
        y = np.random.random([1000, 1])
        m = cflearn.make().fit(x, y)
        cflearn.save(m)
        ms = cflearn.load()
        self.assertTrue(np.allclose(m.predict(x), ms["fcnn"][0].predict(x)))  # type: ignore

    def test_apis4(self) -> None:
        n = 5
        x = np.random.random([1000, 10])
        y = np.random.random([1000, 1])
        result = cflearn.repeat_with(x, y, num_repeat=n)
        pipelines = result.pipelines
        self.assertListEqual(list(pipelines.keys()), ["fcnn"])  # type: ignore
        self.assertEqual(len(list(pipelines.values())[0]), n)  # type: ignore

    def test_distributed1(self) -> None:
        x, y = TabularDataset.iris().xy
        results = cflearn.repeat_with(x, y, num_repeat=3, num_jobs=0)
        patterns = results.patterns["fcnn"]  # type: ignore
        ensemble = cflearn.Ensemble.stacking(patterns)
        patterns_dict = {"fcnn_3": patterns, "fcnn_3_ensemble": ensemble}
        cflearn.evaluate(x, y, metrics=["acc", "auc"], other_patterns=patterns_dict)

    def test_distributed2(self) -> None:
        x = np.random.random([1000, 10])
        y = np.random.random([1000, 1])
        result = cflearn.repeat_with(x, y, models=["linear", "fcnn"], num_repeat=3)
        cflearn.evaluate(x, y, pipelines=result.pipelines)

    def test_distributed3(self) -> None:
        x, y = TabularDataset.iris().xy
        hpo = cflearn.tune_with(
            x,
            y,
            task_type="clf",
            num_repeat=2,
            num_parallel=0,
            num_search=10,
        )
        m = cflearn.make(**hpo.best_param).fit(x, y)
        cflearn.evaluate(x, y, pipelines=m)

    def test_distributed4(self) -> None:
        x = np.random.random([100, 10])
        y = np.random.random([100, 1])
        experiment = cflearn.Experiment()
        data_bundle_folder = experiment.dump_data_bundle(x, y)
        for model in ["linear", "fcnn", "tree_dnn"]:
            experiment.add_task(model=model, data_folder=data_bundle_folder)
        experiment.run_tasks(task_loader=cflearn.task_loader)

    def test_distributed5(self) -> None:
        x1 = np.random.random([100, 10])
        y1 = np.random.random([100, 1])
        x2 = np.random.random([100, 10])
        y2 = np.random.random([100, 1])
        experiment = cflearn.Experiment()
        experiment.add_task(x1, y1)
        experiment.add_task(x2, y2)
        experiment.run_tasks(task_loader=cflearn.task_loader)

    def test_distributed6(self) -> None:
        x = np.random.random([100, 10])
        y = np.random.random([100, 1])
        experiment = cflearn.Experiment()
        data_bundle_folder = experiment.dump_data_bundle(x, y)
        for model in ["linear", "fcnn", "tree_dnn"]:
            experiment.add_task(model=model, data_folder=data_bundle_folder)
        external_path = os.path.join(data_folder, "test_run_sklearn.py")
        run_command = f"{sys.executable} {external_path}"
        experiment.add_task(
            model="svr",
            run_command=run_command,
            data_folder=data_bundle_folder,
        )
        experiment.add_task(
            model="linear_svr",
            run_command=run_command,
            data_folder=data_bundle_folder,
        )
        results = experiment.run_tasks()
        pipelines = {}
        scikit_patterns = {}
        for workplace, workplace_key in zip(results.workplaces, results.workplace_keys):
            model = workplace_key[0]
            if model not in ["svr", "linear_svr"]:
                pipelines[model] = cflearn.task_loader(workplace)
            else:
                model_file = os.path.join(workplace, "sk_model.pkl")
                with open(model_file, "rb") as f:
                    sk_model = pickle.load(f)
                    sk_predict = lambda x_: sk_model.predict(x_).reshape([-1, 1])
                    sk_pattern = cflearn.ModelPattern(predict_method=sk_predict)
                    scikit_patterns[model] = sk_pattern
        cflearn.evaluate(x, y, pipelines=pipelines, other_patterns=scikit_patterns)

    def test_production(self) -> None:
        x, y = TabularDataset.iris().xy
        m = cflearn.make().fit(x, y)
        cflearn.Pack.pack(m, "pack")
        predictor = cflearn.Pack.get_predictor("pack")
        predictor.predict(x)

    def test_iris(self) -> None:
        folder = os.path.join(examples_folder, "iris")
        os.system(f"python {os.path.join(folder, 'iris.py')}")

    def test_titanic(self) -> None:
        folder = os.path.join(examples_folder, "titanic")
        os.system(f"python {os.path.join(folder, 'titanic.py')}")

    def test_op(self) -> None:
        folder = os.path.join(examples_folder, "operations")
        os.system(f"python {os.path.join(folder, 'op.py')}")

    def test_customization1(self) -> None:
        class Foo:
            def __init__(self, dummy_value: float):
                self.dummy = dummy_value

        cflearn.register_config("foo", "one", config={"dummy_value": 1.0})
        cflearn.register_config("foo", "two", config={"dummy_value": 2.0})
        for name, value in zip(["one", "two"], [1.0, 2.0]):
            cfg = cflearn.Configs.get("foo", name)
            config = cfg.pop()
            self.assertEqual(Foo(**config).dummy, value)

    def test_customization2(self) -> None:
        x = np.random.random([10 ** 4, 3])
        y = np.random.random([10 ** 4, 1])

        @cflearn.register_config("svr_meta", "default")
        class SVRMetaConfig(cflearn.Configs):  # type: ignore
            def get_default(self) -> Dict[str, Any]:
                svr = SVR().fit(x, y.ravel())
                return {
                    "support": {"data": ("support", svr.support_)},
                    "intercept": {"data": ("intercept", svr.intercept_)},
                }

        @cflearn.register_extractor("support")
        class Support(Identity):
            def __init__(self, in_flat_dim: int, dimensions: Dimensions, **kwargs: Any):
                # kwargs == svr_meta["support"]
                super().__init__(in_flat_dim, dimensions, **kwargs)
                # so kwargs["data"] == ("support", svr.support_)
                print(kwargs["data"])

        @cflearn.register_extractor("intercept")
        class Intercept(Identity):
            def __init__(self, in_flat_dim: int, dimensions: Dimensions, **kwargs: Any):
                # kwargs == svr_meta["intercept"]
                super().__init__(in_flat_dim, dimensions, **kwargs)
                # so kwargs["data"] == ("intercept", svr.intercept_)
                print(kwargs["data"])

        cflearn.register_model(
            "test_svr",
            pipes=[
                cflearn.PipeInfo(
                    "support",
                    extractor_meta_scope="svr_meta",
                    head="linear",
                ),
                cflearn.PipeInfo(
                    "intercept",
                    extractor_meta_scope="svr_meta",
                    head="linear",
                ),
            ],
        )
        cflearn.make("test_svr").fit(x, y)

    def test_customization3(self) -> None:
        @cflearn.register_model("my_own_linear")
        @cflearn.register_pipe("linear")
        class MyOwnLinear(cflearn.ModelBase):  # type: ignore
            pass

        cflearn.register_model("my_own_linear2", pipes=[cflearn.PipeInfo("linear")])
        cflearn.register_model(
            "brand_new_model",
            pipes=[
                cflearn.PipeInfo("dndf", transform="one_hot_only"),
                cflearn.PipeInfo("linear", transform="one_hot"),
                cflearn.PipeInfo(
                    "linear2",
                    transform="numerical",
                    extractor="identity",
                    head="linear",
                ),
                cflearn.PipeInfo("fcnn", transform="numerical"),
                cflearn.PipeInfo(
                    "fcnn2",
                    transform="embedding_only",
                    extractor="identity",
                    head="fcnn",
                ),
            ],
        )
        m = cflearn.make("brand_new_model")
        if not IS_WINDOWS:
            m.draw("brand_new_model.png", transparent=False)
        numerical = np.random.random([10000, 5])
        categorical = np.random.randint(0, 10, [10000, 5])
        x = np.hstack([numerical, categorical])
        y = np.random.random([10000, 1])
        m.fit(x, y)

    def test_customization4(self) -> None:
        x = np.random.random([10000, 2]) * 2.0
        y = np.prod(x, axis=1, keepdims=True)
        kwargs = {"task_type": "reg", "use_simplify_data": True}
        fcnn = cflearn.make(**kwargs).fit(x, y)  # type: ignore
        cflearn.evaluate(x, y, pipelines=fcnn)

        @cflearn.register_extractor("cross_multiplication")
        class CrossMultiplication(cflearn.ExtractorBase):
            @property
            def out_dim(self) -> int:
                return 1

            def forward(self, net: torch.Tensor) -> torch.Tensor:
                prod = net[..., 0] * net[..., 1]
                return prod.view([-1, 1])

        cflearn.register_config("cross_multiplication", "default", config={})
        cflearn.register_model(
            "multiplication",
            pipes=[cflearn.PipeInfo("linear", extractor="cross_multiplication")],
        )
        mul = cflearn.make("multiplication", **kwargs).fit(x, y)  # type: ignore
        cflearn.evaluate(x, y, pipelines=[fcnn, mul])

        @cflearn.register_head("cross")
        class CrossHead(cflearn.HeadBase):
            def __init__(
                self,
                in_dim: int,
                out_dim: int,
                activation: Optional[str],
                **kwargs: Any,
            ):
                super().__init__(in_dim, out_dim, **kwargs)
                self.cross = CrossBlock(in_dim, residual=False, bias=False)
                if activation is None:
                    self.activation = None
                else:
                    self.activation = Activations.make(activation)
                self.linear = Linear(in_dim, out_dim)

            def forward(self, net: torch.Tensor) -> torch.Tensor:
                net = self.cross(net, net)
                if self.activation is not None:
                    net = self.activation(net)
                return self.linear(net)

        cflearn.register_head_config(
            "cross",
            "default",
            head_config={"activation": None},
        )
        cflearn.register_model("cross", pipes=[cflearn.PipeInfo("cross")])
        cross = cflearn.make("cross", **kwargs).fit(x, y)  # type: ignore
        cflearn.evaluate(x, y, pipelines=[fcnn, mul, cross])

    def test_customization5(self) -> None:
        @cflearn.register_aggregator("prod")
        class Prod(cflearn.AggregatorBase):
            def reduce(
                self,
                outputs: tensor_dict_type,
                **kwargs: Any,
            ) -> tensor_dict_type:
                return {"predictions": outputs["linear"] * outputs["linear2"]}

        cflearn.register_model(
            "prod",
            pipes=[
                cflearn.PipeInfo("linear"),
                cflearn.PipeInfo("linear2", extractor="identity", head="linear"),
            ],
        )
        x = np.random.random([10000, 2]) * 2.0
        y = np.prod(x, axis=1, keepdims=True)
        kwargs = {"task_type": "reg", "use_simplify_data": True}
        prod = cflearn.make("prod", aggregator="prod", **kwargs).fit(x, y)  # type: ignore
        cflearn.evaluate(x, y, pipelines=prod)

    def test_customization6(self) -> None:
        @cflearn.register_loss("to_one")
        class ToOneLoss(cflearn.LossBase):
            def _core(
                self,
                forward_results: tensor_dict_type,
                target: torch.Tensor,
                **kwargs: Any,
            ) -> losses_type:
                return (forward_results["predictions"] - 1.0).abs()

        x = np.random.random([1000, 4])
        y = np.random.random([1000, 1])
        cflearn.make("linear", loss="to_one", metrics="loss").fit(x, y)


if __name__ == "__main__":
    unittest.main()
