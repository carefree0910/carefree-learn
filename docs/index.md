# carefree-learn

`carefree-learn` is a a minimal Automatic Machine Learning (AutoML) solution for tabular datasets based on [PyTorch](https://pytorch.org/){target=_blank}.

#### Why carefree-learn?

`carefree-learn`

+ Provides a [scikit-learn](https://scikit-learn.org/stable/){target=_blank}-like interface with much more 'carefree' usages, including:
    + Automatically deals with data pre-processing.
    + Automatically handles datasets saved in files (.txt, .csv).
    + Supports [Distributed Training](introduction.md#distributed-training), which means hyper-parameter tuning can be very efficient in `carefree-learn`.
+ Includes some brand new techniques which may boost vanilla Neural Network (NN) performances on tabular datasets, including:
    + [`TreeDNN` with `Dynamic Soft Pruning`](https://arxiv.org/pdf/1911.05443.pdf){target=_blank}, which makes NN less sensitive to hyper-parameters. 
    + [`Deep Distribution Regression (DDR)`](https://arxiv.org/pdf/1911.05441.pdf){target=_blank}, which is capable of modeling the entire conditional distribution with one single NN model.
+ Supports many convenient functionality in deep learning, including:
    + Early stopping.
    + Model persistence.
    + Learning rate schedulers.
    + And more...
+ Full utilization of the WIP ecosystem `cf*`, such as:
    + [`carefree-toolkit`](https://github.com/carefree0910/carefree-toolkit){target=_blank}: provides a lot of utility classes & functions which are 'stand alone' and can be leveraged in your own projects.
    + [`carefree-data`](https://github.com/carefree0910/carefree-data){target=_blank}: a lightweight tool to read -> convert -> process **ANY** tabular datasets. It also utilizes [cython](https://cython.org/){target=_blank} to accelerate critical procedures.

From the above, it comes out that `carefree-learn` could be treated as a minimal **Auto**matic **M**achine **L**earning (AutoML) solution for tabular datasets when it is fully utilized. However, this is not built on the sacrifice of flexibility. In fact, the functionality we've mentioned are all wrapped into individual modules in `carefree-learn` and allow users to customize them easily.


## Installation

`carefree-learn` requires Python 3.6 or higher.

### Pre-Installing PyTorch

Please refer to [PyTorch](https://pytorch.org/get-started/locally/){target=_blank}, and it is highly recommended to pre-install PyTorch with conda.

### pip installation

After installing PyTorch, installation of `carefree-learn` would be rather easy:

+ *Tips: if you pre-installed PyTorch with conda, remember to activate the corresponding environment!*

```bash
pip install carefree-learn
```

or

```bash
git clone https://github.com/carefree0910/carefree-learn.git
cd carefree-learn
pip install -e .
```


## Examples

### Quick Start

```python
import cflearn
from cfdata.tabular import TabularDataset

x, y = TabularDataset.iris().xy
m = cflearn.make().fit(x, y)
# Make label predictions
m.predict(x)
# Make probability predictions
m.predict_prob(x)
# Estimate performance
cflearn.estimate(x, y, wrappers=m)

""" Then you will see something like this:

================================================================================================================================
|        metrics         |                       acc                        |                       auc                        |
--------------------------------------------------------------------------------------------------------------------------------
|                        |      mean      |      std       |     score      |      mean      |      std       |     score      |
--------------------------------------------------------------------------------------------------------------------------------
|          fcnn          |    0.946667    |    0.000000    |    0.946667    |    0.993200    |    0.000000    |    0.993200    |
================================================================================================================================

"""

# `carefree-learn` models can be saved easily, into a zip file!
# For example, a `cflearn^_^fcnn.zip` file will be created with this line of code:
cflearn.save(m)
# And loading `carefree-learn` models are easy too!
m = cflearn.load()
# You will see exactly the same result as above!
cflearn.estimate(x, y, wrappers=m)

# `carefree-learn` can also easily fit / predict / estimate directly on files!
# `delim` refers to 'delimiter', and `skip_first` refers to skipping first line or not.
# * Please refer to https://github.com/carefree0910/carefree-data/blob/dev/README.md if you're interested in more details.
""" Suppose we have an 'xor.txt' file with following contents:

0,0,0
0,1,1
1,0,1
1,1,0

"""
m = cflearn.make(delim=",", skip_first=False).fit("xor.txt", x_cv="xor.txt")
cflearn.estimate("xor.txt", wrappers=m)

""" Then you will see something like this:

================================================================================================================================
|        metrics         |                       acc                        |                       auc                        |
--------------------------------------------------------------------------------------------------------------------------------
|                        |      mean      |      std       |     score      |      mean      |      std       |     score      |
--------------------------------------------------------------------------------------------------------------------------------
|          fcnn          |    1.000000    |    0.000000    |    1.000000    |    1.000000    |    0.000000    |    1.000000    |
================================================================================================================================

"""

# When we fit from files, we can predict on either files or lists:
print(m.predict([[0, 0]]))   # [[0]]
print(m.predict([[0, 1]]))   # [[1]]
print(m.predict("xor.txt"))  # [ [0] [1] [1] [0] ]
```

After running the above codes, you might notice that a `_logging` folder is created in your current working directory, which has the following file structure:

```text
-- _logging
 |-- checkpoints
   |-- pipeline_30.pt
   |-- pipeline_31.pt
   |-- pipeline_32.pt
   |-- pipeline_33.pt
   |-- pipeline_34.pt
 |-- 2020-06_26_22-36-17.txt
```

Here, `pipeline_xx.pt` are the top 5 checkpoint files we've obtained during training, and `2020-xxx.txt` is the (simplified) logging file, with contents shown as follows:

```text
| epoch  1   - step   1    | auc : 0.440000 | score : 0.440000 |
| epoch  1   - step   2    | auc : 0.776000 | score : 0.776000 |
| epoch  2   - step   3    | auc : 0.839600 | score : 0.839600 |
| epoch  2   - step   4    | auc : 0.851960 | score : 0.851960 |
| epoch  3   - step   5    | auc : 0.865196 | score : 0.865196 |
...
| epoch  22  - step   44   | auc : 1.000000 | score : 1.000000 |
| epoch  23  - step   45   | auc : 1.000000 | score : 1.000000 |
| epoch  -1  - step   -1   | auc : 1.000000 | score : 1.000000 |
```

!!! info
    The maximum number of checkpoint files can be specified. See **`max_snapshot_num`** in [`cflearn.make`](user-guide/configurations.md#high-level-api) for more details.

### Distributed

In `carefree-learn`, **Distributed Training** doesn't mean training your model on multiple GPUs or multiple machines, because `carefree-learn` focuses on tabular datasets (or, structured datasets) which are often not as large as unstructured datasets. Instead, **Distributed Training** in `carefree-learn` means **training multiple models** at the same time. This is important because:

+ Deep Learning models suffer from randomness, so we need to train multiple models with the same algorithm and calculate the mean / std of the performances to estimate the algorithm's capacity and stability.
+ Ensemble these models (which are trained with the same algorithm) can boost the algorithm's performance without making any changes to the algorithm itself.
+ Parameter searching will be easier & faster.

```python
import cflearn
from cfdata.tabular import TabularDataset

# It is necessary to wrap codes under '__main__' on WINDOWS platform when running distributed codes
if __name__ == '__main__':
    x, y = TabularDataset.iris().xy
    # Notice that 3 fcnn were trained simultaneously with this line of code
    _, patterns = cflearn.repeat_with(x, y, num_repeat=3, num_parallel=3)
    # And it is fairly straight forward to apply stacking ensemble
    ensemble = cflearn.ensemble(patterns)
    patterns_dict = {"fcnn_3": patterns, "fcnn_3_ensemble": ensemble}
    cflearn.estimate(x, y, metrics=["acc", "auc"], other_patterns=patterns_dict)

""" Then you will see something like this:

================================================================================================================================
|        metrics         |                       acc                        |                       auc                        |
--------------------------------------------------------------------------------------------------------------------------------
|                        |      mean      |      std       |     score      |      mean      |      std       |     score      |
--------------------------------------------------------------------------------------------------------------------------------
|         fcnn_3         |    0.937778    |    0.017498    |    0.920280    | -- 0.993911 -- |    0.000274    |    0.993637    |
--------------------------------------------------------------------------------------------------------------------------------
|    fcnn_3_ensemble     | -- 0.953333 -- | -- 0.000000 -- | -- 0.953333 -- |    0.993867    | -- 0.000000 -- | -- 0.993867 -- |
================================================================================================================================

"""
```

!!! info
    You might notice that the best results of each column is 'highlighted' with a pair of '--'.

### Hyper Parameter Optimization (HPO)

```python
import cflearn
from cfdata.tabular import *
 
if __name__ == '__main__':
    x, y = TabularDataset.iris().xy
    # Bayesian Optimization (BO) will be used as default
    hpo = cflearn.tune_with(
        x, y,
        task_type=TaskTypes.CLASSIFICATION,
        num_repeat=2, num_parallel=0, num_search=10
    )
    # We can further train our model with the best hyper-parameters we've obtained:
    m = cflearn.make(**hpo.best_param).fit(x, y)
    cflearn.estimate(x, y, wrappers=m)

""" Then you will see something like this:

~~~  [ info ] Results
================================================================================================================================
|        metrics         |                       acc                        |                       auc                        |
--------------------------------------------------------------------------------------------------------------------------------
|                        |      mean      |      std       |     score      |      mean      |      std       |     score      |
--------------------------------------------------------------------------------------------------------------------------------
|        0659e09f        |    0.943333    |    0.016667    |    0.926667    |    0.995500    |    0.001967    |    0.993533    |
--------------------------------------------------------------------------------------------------------------------------------
|        08a0a030        |    0.796667    |    0.130000    |    0.666667    |    0.969333    |    0.012000    |    0.957333    |
--------------------------------------------------------------------------------------------------------------------------------
|        1962285c        |    0.950000    |    0.003333    |    0.946667    |    0.997467    |    0.000533    |    0.996933    |
--------------------------------------------------------------------------------------------------------------------------------
|        1eb7f2a0        |    0.933333    |    0.020000    |    0.913333    |    0.994833    |    0.003033    |    0.991800    |
--------------------------------------------------------------------------------------------------------------------------------
|        4ed5bb3b        |    0.973333    |    0.013333    |    0.960000    |    0.998733    |    0.000467    |    0.998267    |
--------------------------------------------------------------------------------------------------------------------------------
|        5a652f3c        |    0.953333    | -- 0.000000 -- |    0.953333    |    0.997400    |    0.000133    |    0.997267    |
--------------------------------------------------------------------------------------------------------------------------------
|        82c35e77        |    0.940000    |    0.020000    |    0.920000    |    0.995467    |    0.002133    |    0.993333    |
--------------------------------------------------------------------------------------------------------------------------------
|        a9ef52d0        | -- 0.986667 -- |    0.006667    | -- 0.980000 -- | -- 0.999200 -- | -- 0.000000 -- | -- 0.999200 -- |
--------------------------------------------------------------------------------------------------------------------------------
|        ba2e179a        |    0.946667    |    0.026667    |    0.920000    |    0.995633    |    0.001900    |    0.993733    |
--------------------------------------------------------------------------------------------------------------------------------
|        ec8c0837        |    0.973333    | -- 0.000000 -- |    0.973333    |    0.998867    |    0.000067    |    0.998800    |
================================================================================================================================

~~~  [ info ] Best Parameters
----------------------------------------------------------------------------------------------------
acc  (a9ef52d0) (0.986667 ± 0.006667)
----------------------------------------------------------------------------------------------------
{'optimizer': 'rmsprop', 'optimizer_config': {'lr': 0.005810863965757382}}
----------------------------------------------------------------------------------------------------
auc  (a9ef52d0) (0.999200 ± 0.000000)
----------------------------------------------------------------------------------------------------
{'optimizer': 'rmsprop', 'optimizer_config': {'lr': 0.005810863965757382}}
----------------------------------------------------------------------------------------------------
best (a9ef52d0)
----------------------------------------------------------------------------------------------------
{'optimizer': 'rmsprop', 'optimizer_config': {'lr': 0.005810863965757382}}
----------------------------------------------------------------------------------------------------

~~  [ info ] Results
================================================================================================================================
|        metrics         |                       acc                        |                       auc                        |
--------------------------------------------------------------------------------------------------------------------------------
|                        |      mean      |      std       |     score      |      mean      |      std       |     score      |
--------------------------------------------------------------------------------------------------------------------------------
|          fcnn          |    0.980000    |    0.000000    |    0.980000    |    0.998867    |    0.000000    |    0.998867    |
================================================================================================================================

"""
```

You might notice that:

+ The final results obtained by **HPO** is even better than the stacking ensemble results mentioned above.
+ We search for `optimizer` and `lr` as default. In fact, we can manually passed `params` into `cflearn.tune_with`. If not, then `carefree-learn` will execute following codes:
```python
from cftool.ml.param_utils import *

params = {
    "optimizer": String(Choice(values=["sgd", "rmsprop", "adam"])),
    "optimizer_config": {
        "lr": Float(Exponential(1e-5, 0.1))
    }
}
```

!!! info
    It is also worth mentioning that we can pass file datasets into `cflearn.tune_with` as well. See `tests/usages/test_basic.py` for more details.


## License

`carefree-learn` is MIT licensed, as found in the [`LICENSE`](about/license.md) file.

---
