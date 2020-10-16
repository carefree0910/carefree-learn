# Configurations

*[EMA]: Exponential Moving Average

---

Although it is possible to get a rather good performance with default configurations, performance might be gained easily by specifying configurations with our prior knowledges.

We've mentioned the basic ideas on how to configure `carefree-learn` in [`Introduction`](../introduction.md#configurations), so we will focus on introducing how to actually do it in this page. 

!!! info
    + Notice that configurations listed in this page are algorithm-agnostic.
    + For reference on concepts repeated across the configurations, see [`Terminologies`](../introduction.md#terminologies).


## Design Tenets

There are several common practices for specifying configurations. In many high-level Machine Learning modules (e.g. [scikit-learn](https://scikit-learn.org/stable/){target=_blank}), configurations are directly specified by using args and kwargs to instantiate an object of corresponding algorithm. In `carefree-learn`, however, since we've wrapped many procedures together (in [`Pipeline`](../introduction.md#pipeline)) to provide a more 'carefree' usage, we cannot put all those configurations in the definition of the class because that will be too long and too messy. Instead, we will use one single, shared dict to specify configurations.

There are several advantages by doing so, as listed below:

+ It's much more flexible and easier to extend.
+ It's much easier to reproduce other's work, because a single JSON file will be enough.
+ It's much easier to share configurations between different modules. This is especially helpful in `carefree-learn` because we've tried hard to do elegant abstractions, which lead us to implement many individual modules to handle different problems. In this case, some 'global' information will be hard to access if we don't share configurations.
+ It's possible to define high level APIs which accept args and kwargs for easier usages (e.g. [`cflearn.make`](configurations.md#high-level-api)).


## How to Specify Configurations

There are two ways to specify configurations in `carefree-learn`: directly with a Python dict or indirectly with a JSON file.

### Configure with Python dict

```python
import cflearn

# specify any configurations
config = {"foo": 0, "dummy": 1}
fcnn = cflearn.make(**config)

print(fcnn.config)  # {"foo": 0, "dummy": 1, ...}
```

### Configure with JSON file

In order to use JSON file as configuration, suppose you want to run `my_script.py`, and it contains the following codes:

```python
import cflearn

config = "./configs/basic.json"
increment_config = {"foo": 2}
fcnn = cflearn.make(config=config, increment_config=increment_config)
```

Since `config` is set to `"./configs/basic.json"`, the file structure should be:

```text
-- my_script.py
-- configs
 |-- basic.json
```

Suppose `basic.json` contains following stuffs:

```json
{
    "foo": 0,
    "dummy": 1
}
```

Then the output of `:::python print(fcnn.config)` should be:

```python
{"foo": 2, "dummy": 1, ...}
```

It is OK to get rid of `increment_config`, in which case the configuration will be completely controlled by `basic.json`:

```python
import cflearn

config = "./configs/basic.json"
fcnn = cflearn.make(config=config)

print(fcnn.config)  # {"foo": 0, "dummy": 1, ...}
```


## High Level API

In order to provide out of the box tools, `carefree-learn` implements high level APIs for training, estimating, distributed, HPO, etc. In this section we'll introduce `cflearn.make`, because it contains most of the frequently used configurations, and other APIs' arguments depend on it more or less.

```python
def make(
    model: str = "fcnn",
    *,
    config: general_config_type = None,
    increment_config: general_config_type = None,
    delim: Optional[str] = None,
    task_type: Optional[str] = None,
    skip_first: Optional[bool] = None,
    cv_split: Optional[Union[float, int]] = None,
    min_epoch: Optional[int] = None,
    num_epoch: Optional[int] = None,
    max_epoch: Optional[int] = None,
    batch_size: Optional[int] = None,
    max_snapshot_num: Optional[int] = None,
    clip_norm: Optional[float] = None,
    ema_decay: Optional[float] = None,
    ts_config: Optional[TimeSeriesConfig] = None,
    aggregation: Optional[str] = None,
    aggregation_config: Optional[Dict[str, Any]] = None,
    ts_label_collator_config: Optional[Dict[str, Any]] = None,
    data_config: Optional[Dict[str, Any]] = None,
    read_config: Optional[Dict[str, Any]] = None,
    model_config: Optional[Dict[str, Any]] = None,
    metrics: Optional[Union[str, List[str]]] = None,
    metric_config: Optional[Dict[str, Any]] = None,
    optimizer: Optional[str] = None,
    scheduler: Optional[str] = None,
    optimizer_config: Optional[Dict[str, Any]] = None,
    scheduler_config: Optional[Dict[str, Any]] = None,
    optimizers: Optional[Dict[str, Any]] = None,
    logging_file: Optional[str] = None,
    logging_folder: Optional[str] = None,
    trigger_logging: Optional[bool] = None,
    trial: Optional[Trial] = None,
    tracker_config: Optional[Dict[str, Any]] = None,
    cuda: Optional[Union[int, str]] = None,
    verbose_level: Optional[int] = None,
    use_timing_context: Optional[bool] = None,
    use_tqdm: Optional[bool] = None,
    **kwargs: Any,
) -> Pipeline
```

+ **`model`** [default = `:::python "fcnn"`]
    + Specify which model we're going to use.
    + Currently `carefree-learn` supports:
        + `:::python "fcnn"`, `:::python "nnb"`, `:::python "ndt"`, `:::python "tree_dnn"` and `:::python "ddr"` for basic usages.
        + `:::python "rnn"` and `:::python "transformer"` for time series usages.
+ **`config`** [default = `:::python None`]
    + Specify the configuration.
+ **`increment_config`** [default = `:::python None`]
    + Specify the increment configuration.
+ **`delim`** [default = `:::python None`]
    + Specify the delimiter of the dataset file.
    + Only take effects when we are using file datasets.
+ **`task_type`** [default = `:::python None`]
    + Specify the task type.
+ **`skip_first`** [default = `:::python None`]
    + Specify whether the first row of the dataset file should be skipped.
    + Only take effects when we are using file datasets.
+ **`cv_split`** [default = `:::python None`]
    + Specify the split of the cross validation dataset.
        + If `:::python cv_split < 1`, it will be the 'ratio' comparing to the whole dataset.
        + If `:::python cv_split > 1`, it will be the exact 'size'.
        + If `:::python cv_split == 1`, `:::python cv_split == "ratio" if isinstance(cv_split, float) else "size"`
+ **`min_epoch`** [default = `:::python None`]
    + Specify the minimum number of epoch.
+ **`num_epoch`** [default = `:::python None`]
    + Specify number of epoch. 
    + Notice that in most cases this will not be the final epoch number.
+ **`max_epoch`** [default = `:::python None`]
    + Specify the maximum number of epoch.
+ **`batch_size`** [default = `:::python None`]
    + Specify the number of samples in each batch.
+ **`max_snapshot_num`** [default = `:::python None`]
    + Specify the maximum number of checkpoint files we could save during training.
+ **`clip_norm`** [default = `:::python None`]
    + Given a gradient `g`, and the **`clip_norm`** value, we will normalize `g` so that its L2-norm is less than or equal to **`clip_norm`**.
    + If `:::python 0.`, then no gradient clip will be performed.
+ **`ema_decay`** [default = `:::python None`]
    + When training a model, it is often beneficial to maintain **E**xponential **M**oving **A**verages with a certain decay rate (**`ema_decay`**) of the trained parameters. Evaluations that use averaged parameters sometimes produce significantly better results than the final trained values.
    + If `:::python 0.`, then no EMA will be used.
+ **`ts_config`** [default = `:::python None`]
    + Specify the time series config (experimental).
+ **`aggregation`** [default = `:::python None`]
    + Specify the aggregation used in time series tasks (experimental).
+ **`aggregation_config`** [default = `:::python None`]
    + Specify the configuration of aggregation used in time series tasks (experimental).
+ **`ts_label_collator_config`** [default = `:::python None`]
    + Specify the configuration of the label collator used in time series tasks (experimental).
+ **`data_config`** [default = `:::python None`]
    + kwargs used in [`cfdata.tabular.TabularData`](https://github.com/carefree0910/carefree-data/blob/b80baf0bbe4beb4e6b20c6347714df9ee231e669/cfdata/tabular/wrapper.py#L19){target=_blank}.
+ **`read_config`** [default = `:::python None`]
    + kwargs used in [`cfdata.tabular.TabularData.read`](https://github.com/carefree0910/carefree-data/blob/b80baf0bbe4beb4e6b20c6347714df9ee231e669/cfdata/tabular/wrapper.py#L409){target=_blank}.
+ **`model_config`** [default = `:::python None`]
    + Configurations used in [`Model`](../introduction.md#model).
+ **`metrics`** [default = `:::python None`]
    + Specify which metric(s) are we going to use to monitor our training process
+ **`metric_config`** [default = `:::python None`]
    + Specify the fine grained configurations of metrics. See [`metrics`](../introduction.md#metrics) for more details.
+ **`optimizer`** [default = `:::python None`]
    + Specify which optimizer will be used.
+ **`scheduler`** [default = `:::python None`]
    + Specify which learning rate scheduler will be used.
+ **`optimizer_config`** [default = `:::python None`]
    + Specify optimizer's configuration.
+ **`scheduler_config`** [default = `:::python None`]
    + Specify scheduler's configuration.
+ **`optimizers`** [default = `:::python None`]
    + Specify the fine grained configurations of optimizers and schedulers. See [`optimizers`](../introduction.md#optimizers) for more details.
+ **`logging_file`** [default = `:::python None`]
    + Specify the logging file.
+ **`logging_folder`** [default = `:::python None`]
    + Specify the logging folder.
+ **`trigger_logging`** [default = `:::python None`]
    + Whether log messages into a log file.
+ **`trial`** [default = `:::python None`]
    + `optuna.trial.Trial`, should not be set manually because this argument should only be set in `cflearn.optuna_tune` internally.
+ **`tracker_config`** [default = `:::python None`]
    + Specify the configuration of `cftool.ml.Tracker`. If `:::python None`, then `Tracker` will not be used.
+ **`cuda`** [default = `:::python None`]
    + Specify the working GPU.
+ **`verbose_level`** [default = `:::python None`]
    + Specify the verbose level.
+ **`use_timing_context`** [default = `:::python None`]
    + Whether utilize the `timing_context` or not.
+ **`use_tqdm`** [default = `:::python None`]
    + Whether utilize the `tqdm` progress bar or not.
+ **`kwargs`** [default = `:::python {}`]
    + Other configurations.


---
