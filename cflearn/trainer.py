import os
import re
import json
import math
import torch

import torch.distributed as dist

from typing import Any
from typing import Set
from typing import Dict
from typing import List
from typing import Tuple
from typing import Callable
from typing import Optional
from accelerate import Accelerator
from tqdm.autonotebook import tqdm
from torch.optim import Optimizer
from cftool.misc import print_info
from cftool.misc import print_error
from cftool.misc import safe_execute
from cftool.misc import print_warning
from cftool.misc import shallow_copy_dict
from cftool.misc import sort_dict_by_value
from cftool.array import to_device
from cftool.types import tensor_dict_type
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data.distributed import DistributedSampler

from .schema import ITrainer
from .schema import StepOutputs
from .schema import ILoss
from .schema import TqdmSettings
from .schema import TrainerState
from .schema import IDLModel
from .schema import MetricsOutputs
from .schema import MonitorResults
from .schema import TrainerMonitor
from .schema import IMetric
from .schema import MultipleMetrics
from .schema import TrainerCallback
from .schema import IInference
from .schema import IDataLoader
from .schema import IData
from .schema import PrecisionType
from .schema import TrainerConfig
from .constants import LOSS_KEY
from .constants import PT_PREFIX
from .constants import SCORES_FILE
from .constants import CHECKPOINTS_FOLDER
from .data.utils import TensorBatcher
from .data.pytorch import TorchDataLoader
from .misc.toolkit import summary
from .misc.toolkit import get_ddp_info
from .misc.toolkit import eval_context
from .models.schemas import ModelWithCustomSteps
from .modules.schedulers import WarmupScheduler


class amp_context:
    def __init__(self, enabled: bool):
        if not enabled:
            self.ctx = None
        else:
            self.ctx = torch.cuda.amp.autocast(enabled=True)

    def __enter__(self) -> None:
        if self.ctx is not None:
            self.ctx.__enter__()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.ctx is not None:
            self.ctx.__exit__(exc_type, exc_val, exc_tb)


def get_scores(folder: str) -> Dict[str, float]:
    scores_path = os.path.join(folder, SCORES_FILE)
    if not os.path.isfile(scores_path):
        return {}
    with open(scores_path, "r") as f:
        return json.load(f)


def get_sorted_checkpoints(checkpoint_folder: str) -> List[str]:
    # better checkpoints will be placed earlier,
    #  which means `checkpoints[0]` is the best checkpoint
    scores = get_scores(checkpoint_folder)
    if not scores:
        return []
    return list(sort_dict_by_value(scores, reverse=True).keys())


def weighted_loss_score(config: TrainerConfig, loss_items: Dict[str, float]) -> float:
    if not config.loss_metrics_weights:
        loss = loss_items.get(LOSS_KEY)
        if loss is not None:
            return -loss
        return -sum(loss_items.values()) / len(loss_items)
    score = 0.0
    for k, w in config.loss_metrics_weights.items():
        v = loss_items.get(k)
        if v is None:
            continue
        score -= v * w
    return score


def get_metrics(
    config: TrainerConfig,
    model: IDLModel,
    loss: ILoss,
    metrics: Optional[IMetric],
    inference: IInference,
    loader: IDataLoader,
    *,
    portion: float = 1.0,
    state: Optional[TrainerState] = None,
    forward_kwargs: Optional[Dict[str, Any]] = None,
) -> MetricsOutputs:
    if isinstance(model, ModelWithCustomSteps) and model.custom_evaluate_step:
        use_grad = inference.use_grad_in_predict
        args = config, loader, portion, state, forward_kwargs
        try:
            with eval_context(model, use_grad=use_grad):
                rs = model.evaluate_step(*args)
        except:
            inference.use_grad_in_predict = True
            with eval_context(model, use_grad=True):
                rs = model.evaluate_step(*args)
        return rs
    outputs = inference.get_outputs(
        loader,
        portion=portion,
        state=state,
        metrics=metrics,
        loss=loss if config.use_losses_as_metrics else None,
        return_outputs=False,
        **(forward_kwargs or {}),
    )
    metric_values = {}
    is_positive = {}
    final_scores = []
    loss_items = outputs.loss_items
    metric_outputs = outputs.metric_outputs
    if loss_items is not None:
        metric_values.update(loss_items)
        is_positive.update({k: False for k in loss_items})
        final_scores.append(weighted_loss_score(config, loss_items))
    if metric_outputs is not None:
        metric_values.update(metric_outputs.metric_values)
        is_positive.update(metric_outputs.is_positive)
        final_scores.append(metric_outputs.final_score)
    final_score = sum(final_scores) / len(final_scores)
    return MetricsOutputs(final_score, metric_values, is_positive)


def get_input_sample(loader: IDataLoader, device: torch.device) -> tensor_dict_type:
    sample = next(iter(TensorBatcher(loader, device)))
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            sample[k] = v[:1]
        elif isinstance(v, list):
            sample[k] = [vv[:1] if isinstance(vv, torch.Tensor) else vv for vv in v]
        else:
            sample[k] = v
    return sample


class Trainer(ITrainer):
    model_log_file = "model.txt"
    metrics_log_file = "metrics.txt"
    summary_log_file = "summary.txt"

    def __init__(self, config: TrainerConfig):
        self.config = config
        self.tqdm_settings = safe_execute(TqdmSettings, config.tqdm_settings or {})
        self._current_scheduler_epoch = -1
        self.lr_metrics_updated = False
        self.intermediate: Optional[MetricsOutputs] = None
        self.final_results: Optional[MetricsOutputs] = None
        self.checkpoint_scores: Dict[str, float] = {}

    @property
    def export_config(self) -> Dict[str, Any]:
        ddp_info = get_ddp_info()
        ddp_d = None if ddp_info is None else ddp_info._asdict()
        return {
            "state_config": self.state.config,
            "valid_portion": self.config.valid_portion,
            "mixed_precision": self.config.mixed_precision,
            "clip_norm": self.config.clip_norm,
            "metrics": (
                None
                if self.metrics is None
                else self.metrics.__identifier__
                if not isinstance(self.metrics, MultipleMetrics)
                else [metric.__identifier__ for metric in self.metrics.metrics]
            ),
            "loss_metrics_weights": self.config.loss_metrics_weights,
            "monitors": [monitor.__identifier__ for monitor in self.monitors],
            "callbacks": [callback.__identifier__ for callback in self.callbacks],
            "optimizer_settings": self.config.optimizer_settings,
            "ddp_info": ddp_d,
            "finetune_config": self.config.finetune_config,
            "tqdm_settings": self.tqdm_settings.asdict(),
        }

    @property
    def device(self) -> torch.device:
        return self.accelerator.device

    @property
    def is_local_rank_0(self) -> bool:
        return self.accelerator.is_local_main_process

    @property
    def use_tqdm_in_validation(self) -> bool:
        if not self.is_local_rank_0:
            return False
        if self.tqdm_settings.in_distributed:
            return False
        return self.tqdm_settings.use_tqdm_in_validation or self.state.is_terminate

    @property
    def validation_loader(self) -> IDataLoader:
        return self.valid_loader or self.train_loader_copy

    @property
    def input_sample(self) -> tensor_dict_type:
        return get_input_sample(self.train_loader_copy, self.device)

    @property
    def has_checkpoint_folder(self) -> bool:
        if self.checkpoint_folder is None:
            return False
        return os.path.isdir(self.checkpoint_folder)

    @property
    def workspace(self) -> str:
        return self.config.workspace

    @property
    def checkpoint_folder(self) -> str:
        return os.path.join(self.workspace, CHECKPOINTS_FOLDER)

    # init

    def _init_finetune(self) -> None:
        finetune_config = self.config.finetune_config
        if finetune_config is None:
            return None
        pretrained_ckpt = finetune_config.get("pretrained_ckpt")
        if pretrained_ckpt is None:
            raise ValueError("`rank` should be provided when `finetune` is triggered")
        print_info(f"loading pretrained checkpoint from '{pretrained_ckpt}'...")
        d = torch.load(pretrained_ckpt, map_location=self.device)
        self.model.load_state_dict(d)
        freeze = finetune_config.get("freeze", "")
        freeze_except = finetune_config.get("freeze_except", "")
        if not freeze and not freeze_except:
            return None
        msg_fmt = f"-> {'{}'} parameters will be {'{}'} under '{'{}'}'"
        param_names = []
        if freeze:
            num_frozen = 0
            for name, param in self.model.named_parameters():
                if re.match(freeze, name):
                    num_frozen += 1
                    param.requires_grad_(False)
                    param_names.append(name)
            msg = msg_fmt.format(num_frozen, "frozen", freeze)
        elif freeze_except:
            num_trainable = 0
            for name, param in self.model.named_parameters():
                if not re.match(freeze_except, name):
                    param.requires_grad_(False)
                else:
                    num_trainable += 1
                    param_names.append(name)
            msg = msg_fmt.format(num_trainable, "trainable", freeze_except)
        else:
            msg = "`freeze` & `freeze_except` should not be provided simultaneously"
            raise ValueError(msg)
        print("\n".join(["=" * 100, msg, "-" * 100] + param_names + ["-" * 100]))

    # core

    def post_loss_step(self, loss_dict: tensor_dict_type) -> None:
        # backward
        loss = loss_dict[LOSS_KEY]
        self.accelerator.backward(loss)
        # clip norm
        self.clip_norm_step()
        # optimize
        self.optimizer_step()
        self.scheduler_step()

    def weighted_loss_score(self, loss_items: Dict[str, float]) -> float:
        if not self.config.loss_metrics_weights:
            loss = loss_items.get(LOSS_KEY)
            if loss is not None:
                return -loss
            return -sum(loss_items.values()) / len(loss_items)
        score = 0.0
        for k, w in self.config.loss_metrics_weights.items():
            v = loss_items.get(k)
            if v is None:
                continue
            score -= v * w
        return score

    def clip_norm_step(self) -> None:
        if self.config.clip_norm > 0.0:
            if self.accelerator.sync_gradients:
                self._gradient_norm = self.accelerator.clip_grad_norm_(
                    self.model_for_training.parameters(),
                    max_norm=self.config.clip_norm,
                )

    def optimizer_step(self) -> None:
        for opt in self.optimizers.values():
            opt.step()
        for param in self.model_for_training.parameters():
            param.grad = None

    def scheduler_step(self) -> None:
        if self.config.update_scheduler_per_epoch:
            if self.state.epoch == self._current_scheduler_epoch:
                return
        lr_metric_logged = False
        for key, scheduler in self.schedulers.items():
            if scheduler is not None:
                should_log_lr, kwargs = self._get_scheduler_settings(key, scheduler)
                if should_log_lr or self.config.update_scheduler_per_epoch:
                    lr_metric_logged = True
                    for callback in self.callbacks:
                        callback.log_lr(
                            f"lr-{key}",
                            scheduler.get_last_lr()[0],
                            self.state,
                        )
                scheduler.step(**shallow_copy_dict(kwargs))
        if lr_metric_logged:
            self.lr_metrics_updated = False
        if self.config.update_scheduler_per_epoch:
            self._current_scheduler_epoch = self.state.epoch

    def _get_metrics(
        self,
        *,
        portion: float = 1.0,
        loader: Optional[IDataLoader] = None,
    ) -> MetricsOutputs:
        return get_metrics(
            self.config,
            self.model,
            self.loss,
            self.metrics,
            self.inference,
            loader or self.validation_loader,
            portion=portion,
            state=self.state,
        )

    def _get_scheduler_settings(
        self,
        key: str,
        scheduler: Any,
    ) -> Tuple[bool, Dict[str, Any]]:
        kwargs = {}
        should_log_lr = self.state.should_log_lr
        is_warmup = isinstance(scheduler, WarmupScheduler)
        requires_metric = key in self.schedulers_requires_metric
        if requires_metric and not (is_warmup and not scheduler.finished_warmup):
            if self.intermediate is None:
                kwargs["metrics"] = -math.inf
            else:
                kwargs["metrics"] = self.intermediate.final_score
            should_log_lr &= self.lr_metrics_updated
        return should_log_lr, kwargs

    def _logging_step(self, metrics_outputs: MetricsOutputs) -> None:
        if not self.is_local_rank_0:
            return None
        if self.epoch_tqdm is not None:
            metric_values = shallow_copy_dict(metrics_outputs.metric_values)
            metric_values["score"] = metrics_outputs.final_score
            self.epoch_tqdm.set_postfix(metric_values)
        for callback in self.callbacks:
            callback.log_metrics(metrics_outputs, self.state)
        if self.state.should_log_artifacts:
            for callback in self.callbacks:
                callback.log_artifacts(self)
        if self.state.should_log_metrics_msg:
            for callback in self.callbacks:
                callback.log_metrics_msg(
                    metrics_outputs,
                    self.metrics_log_path,
                    self.state,
                )

    def _monitor_step(self, step_outputs: StepOutputs) -> MonitorResults:
        terminate = False
        save_checkpoint = False
        for monitor in self.monitors:
            monitor.handle_extension(self.state)
        if self.state.should_monitor:
            # get metrics
            if (
                self.valid_loader is not None
                or self.config.recompute_train_losses_in_eval
            ):
                self.intermediate = self._get_metrics(portion=self.config.valid_portion)
            else:
                loss_dict = step_outputs.loss_dict
                is_positive = {k: False for k in loss_dict}
                loss_score = weighted_loss_score(self.config, loss_dict)
                self.intermediate = MetricsOutputs(loss_score, loss_dict, is_positive)
            self.lr_metrics_updated = True
            # logging
            self._logging_step(self.intermediate)
            # check terminate
            if self.state.should_start_snapshot:
                score = self.intermediate.final_score
                if any(monitor.snapshot(score) for monitor in self.monitors):
                    if self.state.can_snapshot:
                        self.state.update_snapshot_epoch()
                        save_checkpoint = True
                if any(monitor.check_terminate(score) for monitor in self.monitors):
                    terminate = True
        return MonitorResults(terminate, save_checkpoint, self.intermediate)

    def _step(self, batch_idx: int, batch: tensor_dict_type) -> StepOutputs:
        batch = to_device(batch, self.device)
        # kwargs
        forward_kw: Dict[str, Any] = {}
        for callback in self.callbacks:
            callback.mutate_train_forward_kwargs(forward_kw, self)
        loss_kw: Dict[str, Any] = {}
        for callback in self.callbacks:
            callback.mutate_train_loss_kwargs(loss_kw, self)
        # allow model defines its own training step
        if (
            isinstance(self.model, ModelWithCustomSteps)
            and self.model.custom_train_step
        ):
            return self.model.train_step(batch_idx, batch, self, forward_kw, loss_kw)
        # forward & loss
        forward_results = self.model.run(batch_idx, batch, self.state, **forward_kw)
        loss_dict = self.loss.run(forward_results, batch, self.state, **loss_kw)
        # post loss step
        self.post_loss_step(loss_dict)
        return StepOutputs(forward_results, {k: v.item() for k, v in loss_dict.items()})

    # api

    def fit(
        self,
        data: IData,
        loss: ILoss,
        model: IDLModel,
        metrics: Optional[IMetric],
        inference: IInference,
        optimizers: Dict[str, Optimizer],
        schedulers: Dict[str, Optional[_LRScheduler]],
        monitors: List[TrainerMonitor],
        callbacks: List[TrainerCallback],
        schedulers_requires_metric: Set[str],
        *,
        config_export_file: Optional[str] = None,
        show_summary: Optional[bool] = None,
        cuda: Optional[str] = None,
    ) -> "Trainer":
        # accelerator
        cpu = False
        if get_ddp_info() is None:
            if cuda is None:
                cpu = True
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = cuda
        if isinstance(self.config.mixed_precision, PrecisionType):
            self.config.mixed_precision = self.config.mixed_precision.value
        self.accelerator = Accelerator(
            cpu=cpu,
            mixed_precision=self.config.mixed_precision,
        )
        # initialize artifact structure
        if self.is_local_rank_0:
            os.makedirs(self.workspace, exist_ok=True)
            self.metrics_log_path = os.path.join(self.workspace, self.metrics_log_file)
            with open(self.metrics_log_path, "w"):
                pass
            os.makedirs(self.checkpoint_folder, exist_ok=True)
        # initialize
        self.model = model
        self.metrics = metrics
        self.monitors = monitors
        self.callbacks = callbacks
        self.schedulers_requires_metric = schedulers_requires_metric
        if self.is_local_rank_0:
            with open(os.path.join(self.workspace, self.model_log_file), "w") as f:
                f.write(str(model))
        self.inference = inference
        # data
        train_loader, valid_loader = data.get_loaders()
        self.train_loader = train_loader
        self.train_loader_copy = train_loader.copy()
        self.train_loader_copy.disable_shuffle()
        self.valid_loader = valid_loader
        if self.config.fixed_epoch is not None:
            num_epoch = max_epoch = self.config.fixed_epoch
        else:
            max_epoch = self.config.max_epoch
            num_epoch = min(max_epoch, self.config.num_epoch)
        self.state = TrainerState(
            train_loader,
            num_epoch=num_epoch,
            max_epoch=max_epoch,
            fixed_steps=self.config.fixed_steps,
            **(self.config.state_config or {}),
        )
        # accelerator prepare
        optimizer_keys = sorted(optimizers)
        (
            self.loss,
            self.model_for_training,
            *optimizer_list,
        ) = self.accelerator.prepare(
            loss,
            model,
            *[optimizers[k] for k in optimizer_keys],
        )
        self.optimizers = {k: optimizer_list[i] for i, k in enumerate(optimizer_keys)}
        self.schedulers = schedulers
        for sch in schedulers.values():
            if sch is not None:
                sch.load_state_dict(sch.state_dict())
        # callback
        self.model.init_with_trainer(self)
        # finetune
        self._init_finetune()
        # verbose
        if show_summary is None:
            show_summary = not self.tqdm_settings.in_distributed
        if self.is_local_rank_0:
            summary_msg = summary(
                self.model,
                to_device(self.input_sample, self.device),
                return_only=not show_summary,
            )
            with open(os.path.join(self.workspace, self.summary_log_file), "w") as f:
                f.write(summary_msg)
        # tqdm
        step_tqdm = None
        self.epoch_tqdm: Optional[tqdm] = None
        if self.is_local_rank_0 and self.tqdm_settings.use_tqdm:
            self.epoch_tqdm = tqdm(
                list(range(self.state.num_epoch)),
                position=self.tqdm_settings.position,
                desc=self.tqdm_settings.desc,
                leave=False,
            )
        # train
        has_ckpt = terminate = False
        if self.is_local_rank_0 and self.epoch_tqdm is None:
            print_info("entered training loop")
        if self.is_local_rank_0 and config_export_file is not None:
            config_export_path = os.path.join(self.workspace, config_export_file)
            with open(config_export_path, "w") as f:
                json.dump(self.export_config, f)
        for callback in self.callbacks:
            callback.before_loop(self)
        while self.state.should_train:
            try:
                self.state.epoch += 1
                if isinstance(self.train_loader, TorchDataLoader):
                    sampler = self.train_loader.loader.sampler
                    if isinstance(sampler, DistributedSampler):
                        sampler.set_epoch(self.state.epoch)
                        if isinstance(self.valid_loader, TorchDataLoader):
                            valid_sampler = self.valid_loader.loader.sampler
                            if isinstance(valid_sampler, DistributedSampler):
                                valid_sampler.set_epoch(self.state.epoch)
                step_iterator = TensorBatcher(self.train_loader, self.device)
                if self.is_local_rank_0 and self.tqdm_settings.use_step_tqdm:
                    step_tqdm = step_iterator = tqdm(
                        step_iterator,
                        total=len(self.train_loader),
                        position=self.tqdm_settings.position + 1,
                        leave=False,
                    )
                for i, batch in enumerate(step_iterator):
                    self.state.step += 1
                    step_outputs = self._step(i, batch)
                    for callback in self.callbacks:
                        callback.after_step(step_outputs, self.state)
                    monitor_results = self._monitor_step(step_outputs)
                    for callback in self.callbacks:
                        callback.after_monitor(monitor_results, self.state)
                    if self.is_local_rank_0 and monitor_results.save_checkpoint:
                        metric_outputs = monitor_results.metric_outputs
                        assert metric_outputs is not None
                        self.save_checkpoint(metric_outputs.final_score)
                    terminate = monitor_results.terminate or self.state.should_terminate
                    if terminate:
                        break
            except KeyboardInterrupt:
                if dist.is_initialized():
                    raise
                print_error("keyboard interrupted")
                terminate = True
            if terminate:
                break
            if self.epoch_tqdm is not None:
                self.epoch_tqdm.total = self.state.num_epoch
                self.epoch_tqdm.update()
        if self.epoch_tqdm is not None:
            if step_tqdm is not None:
                step_tqdm.close()
            self.epoch_tqdm.close()
        # restore
        if self.is_local_rank_0 and self.has_checkpoint_folder:
            if not self.tqdm_settings.in_distributed:
                print_info("rolling back to the best checkpoint")
            has_ckpt = self.restore_checkpoint()
        # finalize
        self.state.set_terminate()
        if self.is_local_rank_0:
            self.final_results = self._get_metrics(portion=self.config.valid_portion)
            self._logging_step(self.final_results)
            if not has_ckpt:
                self.save_checkpoint(self.final_results.final_score)
        for callback in self.callbacks:
            callback.finalize(self)
        return self

    # checkpointing

    def save_checkpoint(
        self,
        score: float,
        folder: Optional[str] = None,
        *,
        no_history: bool = False,
    ) -> None:
        if not self.is_local_rank_0:
            msg = "`save_checkpoint` should not be called when not `is_local_rank_0`"
            raise ValueError(msg)
        if folder is None:
            if self.checkpoint_folder is None:
                msg = "either `folder` or `checkpoint_folder` should be provided"
                raise ValueError(msg)
            folder = self.checkpoint_folder
        state = getattr(self, "state", None)
        pt_file = f"{PT_PREFIX}{-1 if state is None else state.step}.pt"
        if state is None:
            print_warning(
                "`state` is not initialized, "
                "latest model will be saved and the recorded score will always be 0"
            )
            torch.save(self.model.state_dict(), os.path.join(folder, pt_file))
            with open(os.path.join(folder, SCORES_FILE), "w") as f:
                json.dump({pt_file: 0.0}, f)
            return
        # leave top_k snapshots only
        if state.max_snapshot_file > 0:
            checkpoints = get_sorted_checkpoints(folder)
            if len(checkpoints) >= state.max_snapshot_file:
                for file in checkpoints[state.max_snapshot_file - 1 :]:
                    self.checkpoint_scores.pop(file)
                    os.remove(os.path.join(folder, file))
        # pt
        torch.save(self.model.state_dict(), os.path.join(folder, pt_file))
        # scores
        scores = {} if no_history else self.checkpoint_scores
        scores[pt_file] = score
        with open(os.path.join(folder, SCORES_FILE), "w") as f:
            json.dump(sort_dict_by_value(scores, reverse=True), f)

    def restore_checkpoint(
        self,
        folder: Optional[str] = None,
        strict: bool = True,
        state_dict_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> bool:
        if not self.is_local_rank_0:
            msg = "`restore_checkpoint` should not be called when not `is_local_rank_0`"
            raise ValueError(msg)
        if folder is None:
            if self.checkpoint_folder is None:
                msg = "either `folder` or `checkpoint_folder` should be provided"
                raise ValueError(msg)
            folder = self.checkpoint_folder
        checkpoints = get_sorted_checkpoints(folder)
        if not checkpoints:
            if not self.tqdm_settings.in_distributed:
                print_warning(f"no model file found in {folder}")
            return False
        success = False
        for checkpoint in checkpoints:
            model_file = os.path.join(folder, checkpoint)
            if not os.path.isfile(model_file):
                continue
            if not self.tqdm_settings.in_distributed:
                print_info(f"restoring from {model_file}")
            states = torch.load(model_file, map_location=self.device)
            if state_dict_callback is not None:
                state_dict_callback(states)
            self.model.load_state_dict(states, strict)
            success = True
            break
        return success


__all__ = [
    "get_scores",
    "get_sorted_checkpoints",
    "Trainer",
]
