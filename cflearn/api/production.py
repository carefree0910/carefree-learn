import os
import json
import torch

import numpy as np

from abc import ABCMeta
from typing import Any
from typing import Dict
from typing import List
from typing import Type
from typing import Union
from typing import Callable
from typing import Optional
from functools import partial
from cftool.ml import ModelPattern
from cftool.misc import register_core
from cftool.misc import shallow_copy_dict
from cftool.misc import lock_manager
from cftool.misc import Saving
from cftool.misc import LoggingMixin
from mlflow.pyfunc import PythonModel
from mlflow.pyfunc import PythonModelContext

from ..types import data_type
from ..types import np_dict_type
from ..pipeline import Pipeline
from ..protocol import DataProtocol
from ..inference import ONNX
from ..inference import Inference
from ..inference import PreProcessor


class Predictor:
    def __init__(
        self,
        onnx_config: Dict[str, Any],
        preprocessor_folder: str,
        device: Union[str, torch.device] = "cpu",
        *,
        data: Optional[DataProtocol] = None,
        compress: bool = True,
        use_tqdm: bool = False,
    ):
        self.device = device
        preprocessor = PreProcessor.load(
            preprocessor_folder,
            data=data,
            compress=compress,
        )
        self.inference = Inference(
            preprocessor,
            onnx_config=onnx_config,
            use_tqdm=use_tqdm,
        )

    def __str__(self) -> str:
        return f"Predictor({self.inference})"

    __repr__ = __str__

    def predict(
        self,
        x: data_type,
        batch_size: int = 256,
        *,
        contains_labels: bool = False,
        **kwargs: Any,
    ) -> np_dict_type:
        loader = self.inference.preprocessor.make_inference_loader(
            x,
            self.device,
            batch_size,
            is_onnx=self.inference.onnx is not None,
            contains_labels=contains_labels,
        )
        kwargs = shallow_copy_dict(kwargs)
        kwargs["contains_labels"] = contains_labels
        return self.inference.predict(loader, **shallow_copy_dict(kwargs))

    def predict_prob(
        self,
        x: data_type,
        batch_size: int = 256,
        *,
        contains_labels: bool = False,
        **kwargs: Any,
    ) -> np_dict_type:
        kwargs = shallow_copy_dict(kwargs)
        kwargs["returns_probabilities"] = True
        return self.predict(
            x,
            batch_size,
            contains_labels=contains_labels,
            **shallow_copy_dict(kwargs),
        )

    def to_pattern(self, **kwargs: Any) -> ModelPattern:
        kwargs = shallow_copy_dict(kwargs)
        predict = partial(self.predict, **shallow_copy_dict(kwargs))
        kwargs["returns_probabilities"] = True
        predict_prob = partial(self.predict, **shallow_copy_dict(kwargs))
        return ModelPattern(predict_method=predict, predict_prob_method=predict_prob)


class Pack(LoggingMixin):
    def __init__(self, export_folder: str, *, loading: bool):
        if not loading:
            Saving.prepare_folder(self, export_folder)
        self.export_folder = os.path.abspath(export_folder)

    @property
    def onnx_path(self) -> str:
        return os.path.join(self.export_folder, "m.onnx")

    @property
    def onnx_output_names_path(self) -> str:
        return os.path.join(self.export_folder, "output_names.json")

    @property
    def output_probabilities_path(self) -> str:
        return os.path.join(self.export_folder, "output_probabilities.txt")

    @property
    def binary_config_path(self) -> str:
        return os.path.join(self.export_folder, "binary_config.json")

    @property
    def preprocessor_folder(self) -> str:
        return os.path.join(self.export_folder, "preprocessor")

    @classmethod
    def pack(
        cls,
        pipeline: Pipeline,
        export_folder: str,
        *,
        verbose: bool = True,
        compress: bool = True,
        pack_data: bool = True,
        retain_data: bool = False,
        remove_original: bool = True,
        **kwargs: Any,
    ) -> None:
        kwargs = shallow_copy_dict(kwargs)
        kwargs["verbose"] = verbose
        instance = cls(export_folder, loading=False)
        abs_folder = os.path.abspath(export_folder)
        base_folder = os.path.dirname(abs_folder)
        with lock_manager(base_folder, [export_folder]):
            model = pipeline.model
            if model is None:
                raise ValueError("`model` is not generated yet")
            with model.export_context():
                onnx = ONNX(model=model)
                onnx.to_onnx(instance.onnx_path, **shallow_copy_dict(kwargs))
            with open(instance.onnx_output_names_path, "w") as f:
                json.dump(onnx.output_names, f)
            with open(instance.output_probabilities_path, "w") as f:
                f.write(str(int(model.output_probabilities)))
            pipeline.preprocessor.save(
                instance.preprocessor_folder,
                save_data=pack_data,
                retain_data=retain_data,
                compress=False,
            )
            with open(instance.binary_config_path, "w") as f:
                trainer = pipeline.trainer
                inference = trainer.inference
                if inference.binary_threshold is None:
                    trainer._generate_binary_threshold()
                json.dump(inference.binary_config, f)
            if compress:
                Saving.compress(abs_folder, remove_original=remove_original)

    @classmethod
    def pack_multiple(
        cls,
        pipelines: List[Pipeline],
        export_folder: str,
        *,
        verbose: bool = True,
    ) -> None:
        """ This method will simply pack onnx models and won't retain data """
        for i, pipeline in enumerate(pipelines):
            local_export_folder = os.path.join(export_folder, f"m_{i:04d}")
            Pack.pack(
                pipeline,
                local_export_folder,
                verbose=verbose,
                pack_data=False,
                compress=False,
            )

    @classmethod
    def get_predictor(
        cls,
        export_folder: str,
        device: Union[str, torch.device] = "cpu",
        *,
        data: Optional[DataProtocol] = None,
        compress: bool = True,
        use_tqdm: bool = False,
    ) -> Predictor:
        instance = cls(export_folder, loading=True)
        base_folder = os.path.dirname(os.path.abspath(export_folder))
        with lock_manager(base_folder, [export_folder]):
            with Saving.compress_loader(
                export_folder,
                compress,
                remove_extracted=True,
                logging_mixin=instance,
            ):
                with open(instance.onnx_output_names_path, "r") as f:
                    output_names = json.load(f)
                with open(instance.output_probabilities_path, "r") as f:
                    output_probabilities = bool(int(f.read().strip()))
                onnx_config = {
                    "onnx_path": instance.onnx_path,
                    "output_names": output_names,
                    "output_probabilities": output_probabilities,
                }
                predictor = Predictor(
                    onnx_config,
                    instance.preprocessor_folder,
                    device,
                    data=data,
                    compress=False,
                    use_tqdm=use_tqdm,
                )
                with open(instance.binary_config_path, "r") as f:
                    cfg = json.load(f)
                    predictor.inference.inject_binary_config(cfg)
        return predictor


python_models: Dict[str, Type["PythonModelBase"]] = {}


class PythonModelBase(PythonModel, metaclass=ABCMeta):
    """ Should implement self.predictor """

    def predict(
        self,
        context: PythonModelContext,
        model_input: Any,
    ) -> np.ndarray:
        """
        model_inputs should be a pd.DataFrame which looks like:

        |    file    |   other keys   |
        |  "xxx.csv" |  other values  |

        """
        columns = model_input.columns
        data = model_input.values.ravel()
        kwargs = dict(zip(columns[1:], data[1:]))
        return self.predictor.predict(data[0], **kwargs)

    @classmethod
    def register(cls, name: str) -> Callable[[Type], Type]:
        global python_models
        return register_core(name, python_models)


@PythonModelBase.register("pack")
class PackModel(PythonModelBase):
    def load_context(self, context: PythonModelContext) -> None:
        pack_folder = context.artifacts["pack_folder"]
        self.predictor = Pack.get_predictor(pack_folder, compress=False)


@PythonModelBase.register("pipeline")
class PipelineModel(PythonModelBase):
    def load_context(self, context: PythonModelContext) -> None:
        export_folder = context.artifacts["export_folder"]
        self.predictor = Pipeline.load(export_folder, compress=False)


__all__ = ["Pack", "PackModel", "PipelineModel"]
