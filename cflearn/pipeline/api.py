import os
import torch
import shutil

import numpy as np

from abc import abstractmethod
from abc import ABCMeta
from enum import Enum
from tqdm import tqdm
from typing import Any
from typing import Dict
from typing import List
from typing import Type
from typing import Union
from typing import TypeVar
from typing import Callable
from typing import Optional
from tempfile import mkdtemp
from tempfile import TemporaryDirectory
from collections import OrderedDict
from cftool.misc import print_info
from cftool.misc import safe_execute
from cftool.misc import shallow_copy_dict
from cftool.misc import prepare_workspace_from
from cftool.misc import Saving
from cftool.misc import Serializer
from cftool.array import sigmoid
from cftool.array import softmax
from cftool.types import np_dict_type
from cftool.types import tensor_dict_type
from cftool.pipeline import get_workspace

from .core import Block
from .core import Pipeline
from .blocks import SetDefaultsBlock
from .blocks import SetMLDefaultsBlock
from .blocks import PrepareWorkplaceBlock
from .blocks import ExtractStateInfoBlock
from .blocks import BuildLossBlock
from .blocks import BuildModelBlock
from .blocks import BuildMetricsBlock
from .blocks import BuildInferenceBlock
from .blocks import SetTrainerDefaultsBlock
from .blocks import SetMLTrainerDefaultsBlock
from .blocks import BuildMonitorsBlock
from .blocks import BuildCallbacksBlock
from .blocks import BuildOptimizersBlock
from .blocks import BuildTrainerBlock
from .blocks import RecordNumSamplesBlock
from .blocks import ReportBlock
from .blocks import TrainingBlock
from .blocks import SerializeDataBlock
from .blocks import SerializeModelBlock
from .blocks import SerializeOptimizerBlock
from .schema import IEvaluationPipeline
from ..types import sample_weights_type
from ..types import states_callback_type
from ..schema import IData
from ..schema import DLConfig
from ..schema import IDataLoader
from ..schema import MetricsOutputs
from ..trainer import get_scores
from ..trainer import get_metrics
from ..trainer import get_input_sample
from ..trainer import get_sorted_checkpoints
from ..constants import PREDICTIONS_KEY
from ..data.ml import MLData
from ..misc.toolkit import is_local_rank_0
from ..misc.toolkit import get_device


TInferPipeline = TypeVar("TInferPipeline", bound="DLInferencePipeline", covariant=True)


# internal mixins


class _DeviceMixin:
    build_model: BuildModelBlock

    @property
    def device(self) -> torch.device:
        return self.build_model.model.device


class _InferenceMixin:
    focuses: List[Type[Block]]
    is_built: bool

    data: Optional[IData]
    get_block: Callable[[Type[Block]], Any]
    try_get_block: Callable[[Type[Block]], Any]

    # optional callbacks

    def predict_callback(self, results: np_dict_type) -> np_dict_type:
        """changes can happen inplace"""
        return results

    # api

    @property
    def build_model(self) -> BuildModelBlock:
        return self.get_block(BuildModelBlock)

    @property
    def build_inference(self) -> BuildInferenceBlock:
        return self.get_block(BuildInferenceBlock)

    @property
    def serialize_data(self) -> SerializeDataBlock:
        return self.get_block(SerializeDataBlock)

    @property
    def serialize_model(self) -> Optional[SerializeModelBlock]:
        return self.try_get_block(SerializeModelBlock)

    @classmethod
    def build_with(  # type: ignore
        cls: Type[TInferPipeline],
        config: DLConfig,
        states: Optional[tensor_dict_type] = None,
        *,
        data: Optional[IData] = None,
    ) -> TInferPipeline:
        self = cls.init(config)
        # last focus will be the serialization block
        self.build(*[Block.make(b.__identifier__, {}) for b in cls.focuses])
        if states is not None:
            self.build_model.model.load_state_dict(states)
        self.serialize_model.verbose = False
        self.serialize_data.data = self.data = data
        self.is_built = True
        return self

    def to(  # type: ignore
        self: TInferPipeline,
        device: Union[int, str, torch.device],
    ) -> TInferPipeline:
        self.build_model.model.to(device)
        return self

    def predict(
        self,
        loader: IDataLoader,
        *,
        return_classes: bool = False,
        binary_threshold: float = 0.5,
        return_probabilities: bool = False,
        recover_labels: Optional[bool] = None,
        **kwargs: Any,
    ) -> np_dict_type:
        if not self.is_built:
            raise ValueError(
                f"`{self.__class__.__name__}` should be built beforehand, please use "
                "`DLPipelineSerializer.load_inference/evaluation` or `build_with` "
                "to get a built one!"
            )
        kw = shallow_copy_dict(kwargs)
        kw["loader"] = loader
        outputs = safe_execute(self.build_inference.inference.get_outputs, kw)
        results = outputs.forward_results
        # handle predict flags
        if return_classes and return_probabilities:
            raise ValueError(
                "`return_classes` & `return_probabilities`"
                "should not be True at the same time"
            )
        elif not return_classes and not return_probabilities:
            pass
        else:
            predictions = results[PREDICTIONS_KEY]
            if predictions.shape[1] > 2 and return_classes:
                results[PREDICTIONS_KEY] = predictions.argmax(1, keepdims=True)
            else:
                if predictions.shape[1] == 2:
                    probabilities = softmax(predictions)
                else:
                    pos = sigmoid(predictions)
                    probabilities = np.hstack([1.0 - pos, pos])
                if return_probabilities:
                    results[PREDICTIONS_KEY] = probabilities
                else:
                    classes = (probabilities[..., [1]] >= binary_threshold).astype(int)
                    results[PREDICTIONS_KEY] = classes
        # handle recover labels
        if recover_labels is None:
            recover_labels = self.data is not None
        if recover_labels:
            if self.data is None:
                msg = "`recover_labels` is set to `True` but `data` is not provided"
                raise ValueError(msg)
            y = results[PREDICTIONS_KEY]
            results[PREDICTIONS_KEY] = self.data.recover_labels(y)
        # optional callback
        results = self.predict_callback(results)
        # return
        return results


class _EvaluationMixin(_InferenceMixin, IEvaluationPipeline):
    config: DLConfig

    @property
    def build_loss(self) -> BuildLossBlock:
        return self.get_block(BuildLossBlock)

    @property
    def build_metrics(self) -> BuildMetricsBlock:
        return self.get_block(BuildMetricsBlock)

    def evaluate(self, loader: IDataLoader, **kwargs: Any) -> MetricsOutputs:
        return get_metrics(
            self.config,
            self.build_model.model,
            self.build_loss.loss,
            self.build_metrics.metrics,
            self.build_inference.inference,
            loader,
            forward_kwargs=kwargs,
        )


# apis


class PipelineTypes(str, Enum):
    DL_TRAINING = "dl.training"
    ML_TRAINING = "ml.training"
    DL_INFERENCE = "dl.inference"
    DL_EVALUATION = "dl.evaluation"


class TrainingPipeline(
    Pipeline,
    _DeviceMixin,
    _EvaluationMixin,
    metaclass=ABCMeta,
):
    is_built = False

    @property
    @abstractmethod
    def set_defaults_block(self) -> Block:
        pass

    @property
    @abstractmethod
    def set_trainer_defaults_block(self) -> Block:
        pass

    @property
    def build_trainer(self) -> BuildTrainerBlock:
        return self.get_block(BuildTrainerBlock)

    @property
    def building_blocks(self) -> List[Block]:
        return [
            self.set_defaults_block,
            PrepareWorkplaceBlock(),
            ExtractStateInfoBlock(),
            BuildLossBlock(),
            BuildModelBlock(),
            BuildMetricsBlock(),
            BuildInferenceBlock(),
            self.set_trainer_defaults_block,
            BuildMonitorsBlock(),
            BuildCallbacksBlock(),
            BuildOptimizersBlock(),
            BuildTrainerBlock(),
            RecordNumSamplesBlock(),
            ReportBlock(),
            TrainingBlock(),
            SerializeDataBlock(),
            SerializeModelBlock(),
            SerializeOptimizerBlock(),
        ]

    def after_load(self) -> None:
        self.is_built = True
        workspace = prepare_workspace_from("_logs")
        self.config.workspace = workspace

    def prepare(self, data: IData, sample_weights: sample_weights_type = None) -> None:
        self.data = data.set_sample_weights(sample_weights)
        self.training_workspace = self.config.workspace
        if not self.is_built:
            self.build(*self.building_blocks)
            self.is_built = True
        else:
            for block in self.blocks:
                block.training_workspace = self.training_workspace

    def fit(
        self,
        data: IData,
        *,
        sample_weights: sample_weights_type = None,
        cuda: Optional[Union[int, str]] = None,
    ) -> "TrainingPipeline":
        # build pipeline
        self.prepare(data, sample_weights)
        # check rank 0
        workspace = self.config.workspace if is_local_rank_0() else None
        # save data info
        if workspace is not None:
            Serializer.save(
                os.path.join(workspace, SerializeDataBlock.package_folder),
                data,
                save_npd=False,
            )
        # run pipeline
        self.run(data, cuda=cuda)
        # save pipeline
        if workspace is not None:
            pipeline_folder = DLPipelineSerializer.pipeline_folder
            DLPipelineSerializer.save(self, os.path.join(workspace, pipeline_folder))
        # return
        return self


@Pipeline.register(PipelineTypes.DL_TRAINING)
class DLTrainingPipeline(TrainingPipeline):
    @property
    def set_defaults_block(self) -> Block:
        return SetDefaultsBlock()

    @property
    def set_trainer_defaults_block(self) -> Block:
        return SetTrainerDefaultsBlock()


@Pipeline.register(PipelineTypes.ML_TRAINING)
class MLTrainingPipeline(TrainingPipeline):
    data: MLData

    @property
    def set_defaults_block(self) -> Block:
        return SetMLDefaultsBlock()

    @property
    def set_trainer_defaults_block(self) -> Block:
        return SetMLTrainerDefaultsBlock()


@Pipeline.register(PipelineTypes.DL_INFERENCE)
class DLInferencePipeline(Pipeline, _DeviceMixin, _InferenceMixin):
    is_built = False

    focuses = [
        BuildModelBlock,
        BuildInferenceBlock,
        SerializeDataBlock,
        SerializeModelBlock,
    ]

    def after_load(self) -> None:
        self.is_built = True
        self.data = self.serialize_data.data
        if self.serialize_model is not None:
            self.serialize_model.verbose = False


@Pipeline.register(PipelineTypes.DL_EVALUATION)
class DLEvaluationPipeline(DLInferencePipeline, _EvaluationMixin):
    focuses = [
        BuildLossBlock,
        BuildModelBlock,
        BuildMetricsBlock,
        BuildInferenceBlock,
        SerializeDataBlock,
        SerializeModelBlock,
    ]


class PackType(str, Enum):
    TRAINING = "training"
    INFERENCE = "inference"
    EVALUATION = "evaluation"


class DLPipelineSerializer:
    id_file = "id.txt"
    config_file = "config.json"
    blocks_file = "blocks.json"
    pipeline_folder = "pipeline"

    # api

    @classmethod
    def save(cls, pipeline: Pipeline, folder: str, *, compress: bool = False) -> None:
        original_folder = None
        if compress:
            original_folder = folder
            folder = mkdtemp()
        Serializer.save(folder, pipeline)
        for block in pipeline.blocks:
            block.save_extra(os.path.join(folder, block.__identifier__))
        if compress and original_folder is not None:
            abs_folder = os.path.abspath(folder)
            abs_original = os.path.abspath(original_folder)
            Saving.compress(abs_folder)
            shutil.move(f"{abs_folder}.zip", f"{abs_original}.zip")

    @classmethod
    def pack(
        cls,
        workspace: str,
        export_folder: str,
        *,
        pack_type: PackType = PackType.INFERENCE,
        compress: bool = True,
    ) -> None:
        if pack_type == PackType.TRAINING:
            swap_id = None
            focuses = None
            excludes = [PrepareWorkplaceBlock]
        elif pack_type == PackType.INFERENCE:
            swap_id = DLInferencePipeline.__identifier__
            focuses = DLInferencePipeline.focuses
            excludes = None
        elif pack_type == PackType.EVALUATION:
            swap_id = DLEvaluationPipeline.__identifier__
            focuses = DLEvaluationPipeline.focuses
            excludes = None
        else:
            raise ValueError(f"unrecognized `pack_type` '{pack_type}' occurred")
        pipeline_folder = os.path.join(workspace, cls.pipeline_folder)
        pipeline = cls._load(
            pipeline_folder,
            swap_id=swap_id,
            focuses=focuses,
            excludes=excludes,
        )
        cls.save(pipeline, export_folder, compress=compress)

    @classmethod
    def pack_and_load_inference(cls, workplace: str) -> DLInferencePipeline:
        with TemporaryDirectory() as tmp_folder:
            cls.pack(
                workplace,
                export_folder=tmp_folder,
                pack_type=PackType.INFERENCE,
                compress=False,
            )
            return cls.load_inference(tmp_folder)

    @classmethod
    def pack_onnx(
        cls,
        workplace: str,
        export_file: str = "model.onnx",
        dynamic_axes: Optional[Union[List[int], Dict[int, str]]] = None,
        *,
        input_sample: Optional[tensor_dict_type] = None,
        loader_sample: Optional[IDataLoader] = None,
        opset: int = 11,
        simplify: bool = True,
        num_samples: Optional[int] = None,
        verbose: bool = True,
        **kwargs: Any,
    ) -> DLInferencePipeline:
        if input_sample is None and loader_sample is None:
            msg = "either `input_sample` or `loader_sample` should be provided"
            raise ValueError(msg)
        m = cls.pack_and_load_inference(workplace)
        model = m.build_model.model
        if input_sample is None:
            input_sample = get_input_sample(loader_sample, get_device(model))  # type: ignore
        model.to_onnx(
            export_file,
            input_sample,
            dynamic_axes,
            opset=opset,
            simplify=simplify,
            num_samples=num_samples,
            verbose=verbose,
            **kwargs,
        )
        return m

    @classmethod
    def pack_scripted(
        cls,
        workplace: str,
        export_file: str = "model.pt",
    ) -> DLInferencePipeline:
        m = cls.pack_and_load_inference(workplace)
        model = torch.jit.script(m.build_model.model)
        torch.jit.save(model, export_file)
        return m

    @classmethod
    def fuse_inference(
        cls,
        src_folders: List[str],
        *,
        cuda: Optional[str] = None,
        num_picked: Optional[Union[int, float]] = None,
        states_callback: states_callback_type = None,
    ) -> DLInferencePipeline:
        return cls._fuse_multiple(
            src_folders,
            PackType.INFERENCE,
            cuda,
            num_picked,
            states_callback,
        )

    @classmethod
    def fuse_evaluation(
        cls,
        src_folders: List[str],
        *,
        cuda: Optional[str] = None,
        num_picked: Optional[Union[int, float]] = None,
        states_callback: states_callback_type = None,
    ) -> DLEvaluationPipeline:
        return cls._fuse_multiple(
            src_folders,
            PackType.EVALUATION,
            cuda,
            num_picked,
            states_callback,
        )

    @classmethod
    def load_training(cls, folder: str) -> TrainingPipeline:
        return cls._load(folder, swap_id=DLTrainingPipeline.__identifier__)

    @classmethod
    def load_inference(cls, folder: str) -> DLInferencePipeline:
        return cls._load_inference(folder)

    @classmethod
    def load_evaluation(cls, folder: str) -> DLEvaluationPipeline:
        return cls._load_evaluation(folder)

    # internal

    @classmethod
    def _load(
        cls,
        folder: str,
        *,
        swap_id: Optional[str] = None,
        focuses: Optional[List[Type[Block]]] = None,
        excludes: Optional[List[Type[Block]]] = None,
    ) -> Pipeline:
        with get_workspace(folder) as workspace:
            # handle info
            info = Serializer.load_info(workspace)
            if focuses is not None or excludes is not None:
                if focuses is None:
                    focuses_set = None
                else:
                    focuses_set = {b.__identifier__ for b in focuses}
                block_types = info["blocks"]
                if focuses_set is not None:
                    block_types = [b for b in block_types if b in focuses_set]
                    left = sorted(focuses_set - set(block_types))
                    if left:
                        raise ValueError(
                            "following blocks are specified in `focuses` "
                            f"but not found in the loaded blocks: {', '.join(left)}"
                        )
                if excludes is not None:
                    excludes_set = {b.__identifier__ for b in excludes}
                    block_types = [b for b in block_types if b not in excludes_set]
                info["blocks"] = block_types
            # load
            pipeline = Serializer.load_empty(workspace, Pipeline, swap_id=swap_id)
            pipeline.serialize_folder = workspace
            if info is None:
                info = Serializer.load_info(workspace)
            pipeline.from_info(info)
            for block in pipeline.blocks:
                block.load_from(os.path.join(workspace, block.__identifier__))
            pipeline.after_load()
        return pipeline

    @classmethod
    def _load_inference(
        cls,
        folder: str,
        excludes: Optional[List[Type[Block]]] = None,
    ) -> DLInferencePipeline:
        return cls._load(
            folder,
            swap_id=DLInferencePipeline.__identifier__,
            focuses=DLInferencePipeline.focuses,
            excludes=excludes,
        )

    @classmethod
    def _load_evaluation(
        cls,
        folder: str,
        excludes: Optional[List[Type[Block]]] = None,
    ) -> DLEvaluationPipeline:
        return cls._load(
            folder,
            swap_id=DLEvaluationPipeline.__identifier__,
            focuses=DLEvaluationPipeline.focuses,
            excludes=excludes,
        )

    @classmethod
    def _fuse_multiple(
        cls,
        src_folders: List[str],
        pack_type: PackType,
        cuda: Optional[str] = None,
        num_picked: Optional[Union[int, float]] = None,
        states_callback: states_callback_type = None,
    ) -> DLInferencePipeline:
        if pack_type == PackType.TRAINING:
            raise ValueError("should not pack to training pipeline when fusing")
        # get num picked
        num_total = num_repeat = len(src_folders)
        if num_picked is not None:
            if isinstance(num_picked, float):
                if num_picked < 0.0 or num_picked > 1.0:
                    raise ValueError("`num_picked` should ∈ [0, 1] when set to float")
                num_picked = round(num_total * num_picked)
            if num_picked < 1:
                raise ValueError("calculated `num_picked` should be at least 1")
            scores = []
            for i, folder in enumerate(src_folders):
                ckpt_folder = os.path.join(folder, SerializeModelBlock.__identifier__)
                folder_scores = get_scores(ckpt_folder)
                scores.append(max(folder_scores.values()))
            scores_array = np.array(scores)
            picked_indices = np.argsort(scores)[::-1][:num_picked]
            src_folders = [src_folders[i] for i in picked_indices]
            original_score = scores_array.mean().item()
            picked_score = scores_array[picked_indices].mean().item()
            print_info(
                f"picked {num_picked} / {num_total}, "
                f"score: {original_score} -> {picked_score}"
            )
            num_repeat = num_picked
        # get empty pipeline
        with get_workspace(src_folders[0], force_new=True) as workspace:
            info = Serializer.load_info(workspace)
            config: DLConfig = DLConfig.from_pack(info["config"])
            config.num_repeat = num_repeat
            info["config"] = config.to_pack().asdict()
            Serializer.save_info(workspace, info=info)
            fn = (
                cls._load_inference
                if pack_type == PackType.INFERENCE
                else cls._load_evaluation
            )
            # avoid loading model because the ensembled model has different states
            m = fn(workspace, excludes=[SerializeModelBlock])
            # but we need to build the SerializeModelBlock again for further save/load
            b_serialize_model = SerializeModelBlock()
            b_serialize_model.verbose = False
            m.build(b_serialize_model)
        # merge state dict
        merged_states: OrderedDict[str, torch.Tensor] = OrderedDict()
        for i, folder in enumerate(tqdm(src_folders, desc="fuse")):
            with get_workspace(folder) as i_folder:
                ckpt_folder = os.path.join(i_folder, SerializeModelBlock.__identifier__)
                checkpoints = get_sorted_checkpoints(ckpt_folder)
                checkpoint_path = os.path.join(ckpt_folder, checkpoints[0])
                states = torch.load(checkpoint_path, map_location=cuda)
            current_keys = list(states.keys())
            for k, v in list(states.items()):
                states[f"ms.{i}.{k}"] = v
            for k in current_keys:
                states.pop(k)
            if states_callback is not None:
                states = states_callback(m, states)
            merged_states.update(states)
        # load state dict
        model = m.build_model.model
        model.to(cuda)
        model.load_state_dict(merged_states)
        return m


__all__ = [
    "IEvaluationPipeline",
    "PipelineTypes",
    "TrainingPipeline",
    "DLTrainingPipeline",
    "MLTrainingPipeline",
    "DLInferencePipeline",
    "DLEvaluationPipeline",
    "PackType",
    "DLPipelineSerializer",
]
