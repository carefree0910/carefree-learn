import os
import json
import torch

import numpy as np

from typing import Any
from typing import List
from typing import Union
from typing import Optional
from cftool.ml import EnsemblePattern
from cftool.misc import lock_manager
from cftool.misc import Saving
from cftool.misc import LoggingMixin

from .ensemble import ensemble
from ..misc.toolkit import compress_zip
from ..pipeline.core import Pipeline
from ..pipeline.inference import ONNX
from ..pipeline.inference import Predictor


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
    def binary_config_path(self) -> str:
        return os.path.join(self.export_folder, "binary_config.json")

    @property
    def preprocessor_folder(self) -> str:
        return os.path.join(self.export_folder, "preprocessor")

    # TODO : check binary threshold, BN, EMA, Dropout, etc.

    @classmethod
    def pack(
        cls,
        pipeline: Pipeline,
        export_folder: str,
        *,
        compress: bool = True,
        remove_original: bool = True,
    ) -> None:
        instance = cls(export_folder, loading=False)
        onnx = ONNX(model=pipeline.model).to_onnx(instance.onnx_path)
        with open(instance.onnx_output_names_path, "w") as f:
            json.dump(onnx.output_names, f)
        pipeline.preprocessor.save(instance.preprocessor_folder)
        with open(instance.binary_config_path, "w") as f:
            json.dump(pipeline.trainer.inference.binary_config, f)
        if compress:
            compress_zip(export_folder, remove_original=remove_original)

    @classmethod
    def get_predictor(
        cls,
        export_folder: str,
        device: Union[str, torch.device] = "cpu",
        *,
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
                    onnx_config = {
                        "onnx_path": instance.onnx_path,
                        "output_names": json.load(f),
                    }
                predictor = Predictor(
                    onnx_config,
                    instance.preprocessor_folder,
                    device,
                    use_tqdm=use_tqdm,
                )
                with open(instance.binary_config_path, "r") as f:
                    predictor.inference.inject_binary_config(json.load(f))
        return predictor

    @staticmethod
    def ensemble(
        predictors: List[Predictor],
        weights: Optional[np.ndarray],
        **kwargs: Any,
    ) -> EnsemblePattern:
        patterns = [predictor.to_pattern(**kwargs) for predictor in predictors]
        return ensemble(patterns, pattern_weights=weights)


__all__ = ["Pack"]
