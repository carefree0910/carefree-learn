# Configurations

*[EMA]: Exponential Moving Average

---

Although it is possible to get a rather good performance with default configurations, performance might be gained easily by specifying configurations with our prior knowledges.

We've mentioned the basic ideas on how to configure `carefree-learn` in [`Introduction`](../introduction.md#configurations), so we will focus on introducing how to actually do it in this page. 

!!! info
    + Notice that configurations listed in this page are algorithm-agnostic.
    + For reference on concepts repeated across the configurations, see [`Terminologies`](../introduction.md#terminologies).


## Design Tenets

There are several common practices for specifying configurations. In many high-level Machine Learning modules (e.g. [scikit-learn](https://scikit-learn.org/stable/){target=_blank}), configurations are directly specified by using args and kwargs to instantiate an object of corresponding algorithm. In `carefree-learn`, however, since we've wrapped many procedures together (in [`Wrapper`](../introduction.md#wrapper)) to provide a more 'carefree' usage, we cannot put all those configurations in the definition of the class because that will be too long and too messy. Instead, we will use one single, shared dict to specify configurations.

There are several advantages by doing so, as listed below:

+ It's much more flexible and easier to extend.
+ It's much easier to reproduce other's work, because a single JSON file will be enough.
+ It's much easier to share configurations between different modules. This is especially helpful in `carefree-learn` because we've tried hard to do elegant abstractions, which lead us to implement many individual modules to handle different problems. In this case, some 'global' information will be hard to access if we don't share configurations.
+ It's possible to define high level APIs which accept args and kwargs for easier usages (e.g. [`cflearn.make`](configurations.md#high-level-api)).


## How to Specify Configurations

There are two ways to specify configurations in `carefree-learn`: directly with a Python dict or indirectly with a JSON file.

### Configure with Python dict

We've left `config` & `increment_config` kwargs for you to specify any configurations directly with Python dict:

```python
import cflearn

# specify any configurations
config = {"foo": 0, "dummy": 1}
increment_config = {"foo": 2}
fcnn = cflearn.Wrapper(config, increment_config=increment_config)

print(fcnn.config)  # {"foo": 0, "dummy": 1, ...}
```

### Configure with JSON file

In order to use JSON file as configuration, suppose you want to run `my_script.py`, and it contains the following codes:

```python
import cflearn

config = "./configs/basic.json"
increment_config = {"foo": 2}
fcnn = cflearn.Wrapper(config, increment_config=increment_config)
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
fcnn = cflearn.Wrapper(config)

print(fcnn.config)  # {"foo": 0, "dummy": 1, ...}
```


## High Level API

In order to provide out of the box tools, `carefree-learn` implements high level APIs for training, estimating, distributed, HPO, etc. In this section we'll introduce `cflearn.make`, because it contains most of the frequently used configurations, and other APIs' arguments depend on it more or less.

```python
def make(model: str = "fcnn",
         *,
         delim: str = None,
         skip_first: bool = None,
         cv_split: float = 0.1,
         min_epoch: int = None,
         num_epoch: int = None,
         max_epoch: int = None,
         batch_size: int = None,
         max_snapshot_num: int = None,
         clip_norm: float = None,
         ema_decay: float = None,
         data_config: Dict[str, Any] = None,
         read_config: Dict[str, Any] = None,
         model_config: Dict[str, Any] = None,
         metrics: Union[str, List[str]] = None,
         metric_config: Dict[str, Any] = None,
         optimizer: str = None,
         scheduler: str = None,
         optimizer_config: Dict[str, Any] = None,
         scheduler_config: Dict[str, Any] = None,
         optimizers: Dict[str, Any] = None,
         logging_file: str = None,
         logging_folder: str = None,
         trigger_logging: bool = None,
         cuda: Union[int, str] = 0,
         verbose_level: int = 2,
         use_tqdm: bool = True,
         **kwargs) -> Wrapper
```

+ **`model`** [default = `:::python "fcnn"`]
    + Specify which model we're going to use.
    + Currently `carefree-learn` supports `:::python "fcnn"`, `:::python "nnb"`, `:::python "ndt"`, `:::python "tree_dnn"` and `:::python "ddr"`.
+ **`delim`** [default = `:::python None`]
    + Specify the delimiter of the dataset file.
    + Only take effects when we are using file datasets.
+ **`skip_first`** [default = `:::python None`]
    + Specify whether the first row of the dataset file should be skipped.
    + Only take effects when we are using file datasets.
+ **`cv_split`** [default = `:::python 0.1`]
    + Specify the split of the cross validation dataset.
        + If `:::python cv_split < 1`, it will be the 'ratio' comparing to the whole dataset.
        + If `:::python cv_split > 1`, it will be the exact 'size'.
        + If `:::python cv_split == 1`, `:::python cv_split == "ratio" if isinstance(cv_split, float) else "size"`
+ **`min_epoch`** [default = `:::python 0`]
    + Specify the minimum number of epoch.
+ **`num_epoch`** [default = `:::python 40`]
    + Specify number of epoch. 
    + Notice that in most cases this will not be the final epoch number.
+ **`max_epoch`** [default = `:::python 200`]
    + Specify the maximum number of epoch.
+ **`batch_size`** [default = `:::python 128`]
    + Specify the number of samples in each batch.
+ **`max_snapshot_num`** [default = `:::python 5`]
    + Specify the maximum number of checkpoint files we could save during training.
+ **`clip_norm`** [default = `:::python 0.`]
    + Given a gradient `g`, and the **`clip_norm`** value, we will normalize `g` so that its L2-norm is less than or equal to **`clip_norm`**.
    + If `:::python 0.`, then no gradient clip will be performed.
+ **`ema_decay`** [default = `:::python 0.`]
    + When training a model, it is often beneficial to maintain **E**xponential **M**oving **A**verages with a certain decay rate (**`ema_decay`**) of the trained parameters. Evaluations that use averaged parameters sometimes produce significantly better results than the final trained values.
    + If `:::python 0.`, then no EMA will be used.
+ **`data_config`** [default = `:::python {}`]
    + kwargs used in [`cfdata.tabular.TabularData`](https://github.com/carefree0910/carefree-data/blob/b80baf0bbe4beb4e6b20c6347714df9ee231e669/cfdata/tabular/wrapper.py#L19){target=_blank}.
+ **`read_config`** [default = `:::python {}`]
    + kwargs used in [`cfdata.tabular.TabularData.read`](https://github.com/carefree0910/carefree-data/blob/b80baf0bbe4beb4e6b20c6347714df9ee231e669/cfdata/tabular/wrapper.py#L409){target=_blank}.
+ **`model_config`** [default = `:::python {}`]
    + Configurations used in [`Model`](../introduction.md#model).
+ **`metrics`** [default = `:::python None`]
    + Specify which metric(s) are we going to use to monitor our training process
+ **`metric_config`** [default = `:::python {}`]
    + Specify the fine grained configurations of metrics. See [`metrics`](../introduction.md#metrics) for more details.
+ **`optimizer`** [default = `:::python "adam"`]
    + Specify which optimizer will be used.
+ **`scheduler`** [default = `:::python "plateau"`]
    + Specify which learning rate scheduler will be used.
+ **`optimizer_config`** [default = `:::python {}`]
    + Specify optimizer's configuration.
+ **`scheduler_config`** [default = `:::python {}`]
    + Specify scheduler's configuration.
+ **`optimizers`** [default = `:::python {}`]
    + Specify the fine grained configurations of optimizers and schedulers. See [`optimizers`](../introduction.md#optimizers) for more details.
+ **`logging_file`** [default = `:::python f"{timestamp()}.log"`]
    + Specify the logging file.
+ **`logging_folder`** [default = `:::python f"_logging/{model}"`]
    + Specify the logging folder.
+ **`trigger_logging`** [default = `:::python False`]
    + Whether log messages into a log file.
+ **`cuda`** [default = `:::python 0`]
    + Specify the working GPU.
+ **`verbose_level`** [default = `:::python 2`]
    + Specify the verbose level.
+ **`use_tqdm`** [default = `:::python True`]
    + Whether utilize the `tqdm` progress bar or not.
+ **`kwargs`** [default = `:::python {}`]
    + Other configurations.


---
