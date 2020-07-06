# carefree-learn

`carefree-learn` is a minimal Automatic Machine Learning (AutoML) solution for tabular datasets based on [PyTorch](https://pytorch.org/).

#### Why carefree-learn?

`carefree-learn`

+ Provides a [scikit-learn](https://scikit-learn.org/stable/)-like interface with much more 'carefree' usages, including:
    + Automatically deals with data pre-processing.
    + Automatically handles datasets saved in files (.txt, .csv).
    + Supports Distributed Training, which means hyper-parameter tuning can be very efficient in `carefree-learn`.
+ Includes some brand new techniques which may boost vanilla Neural Network (NN) performances on tabular datasets, including:
    + [`TreeDNN` with `Dynamic Soft Pruning`](https://arxiv.org/pdf/1911.05443.pdf), which makes NN less sensitive to hyper-parameters. 
    + [`Deep Distribution Regression (DDR)`](https://arxiv.org/pdf/1911.05441.pdf), which is capable of modeling the entire conditional distribution with one single NN model.
+ Supports many convenient functionality in deep learning, including:
    + Early stopping.
    + Model persistence.
    + Learning rate schedulers.
    + And more...
+ Full utilization of the WIP ecosystem `cf*`, such as:
    + [`carefree-toolkit`](https://github.com/carefree0910/carefree-toolkit): provides a lot of utility classes & functions which are 'stand alone' and can be leveraged in your own projects.
    + [`carefree-data`](https://github.com/carefree0910/carefree-data): a lightweight tool to read -> convert -> process **ANY** tabular datasets. It also utilizes [cython](https://cython.org/) to accelerate critical procedures.

From the above, it comes out that `carefree-learn` could be treated as a minimal **Auto**matic **M**achine **L**earning (AutoML) solution for tabular datasets when it is fully utilized. However, this is not built on the sacrifice of flexibility. In fact, the functionality we've mentioned are all wrapped into individual modules in `carefree-learn` and allow users to customize them easily.


## Installation

`carefree-learn` requires Python 3.6 or higher.

### Pre-Installing PyTorch

Please refer to [PyTorch](https://pytorch.org/get-started/locally/), and it is highly recommended to pre-install PyTorch with conda.

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

For detailed information, please visit the [documentation](http://carefree0910.me/carefree-learn-doc).

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

It is also worth mention that we can pass file datasets into `cflearn.tune_with` as well. See `tests/usages/test_basic.py` for more details.


## License

`carefree-learn` is MIT licensed, as found in the [LICENSE](https://github.com/carefree0910/carefree-learn/blob/master/LICENSE) file.
