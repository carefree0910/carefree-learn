# carefree-learn

`carefree-learn` is a a minimal Automatic Machine Learning (AutoML) solution for tabular datasets based on [PyTorch](https://pytorch.org/).

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

=============================================================================
|        metrics         |                       auc                        |
-----------------------------------------------------------------------------
|                        |      mean      |      std       |     score      |
-----------------------------------------------------------------------------
|          fcnn          |    0.992667    |    0.000000    |    0.992667    |
=============================================================================

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

=============================================================================
|        metrics         |                       auc                        |
-----------------------------------------------------------------------------
|                        |      mean      |      std       |     score      |
-----------------------------------------------------------------------------
|          fcnn          |    1.000000    |    0.000000    |    1.000000    |
=============================================================================

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
    cflearn.estimate(x, y, metrics=["acc", "auc"], wrappers=m)

""" Then you will see something like this:

~~~  [ info ] Results
================================================================================================================================
|        metrics         |                       acc                        |                       auc                        |
--------------------------------------------------------------------------------------------------------------------------------
|                        |      mean      |      std       |     score      |      mean      |      std       |     score      |
--------------------------------------------------------------------------------------------------------------------------------
|        00ad4e44        | -- 0.943333 -- |    0.030000    | -- 0.913333 -- | -- 0.997467 -- |    0.001067    | -- 0.996400 -- |
--------------------------------------------------------------------------------------------------------------------------------
|        0b5fb5ed        |    0.496667    |    0.163333    |    0.333333    |    0.757367    |    0.046567    |    0.710800    |
--------------------------------------------------------------------------------------------------------------------------------
|        4856a5ea        |    0.806667    |    0.166667    |    0.640000    |    0.870933    |    0.127600    |    0.743333    |
--------------------------------------------------------------------------------------------------------------------------------
|        6476e332        |    0.723333    |    0.056667    |    0.666667    |    0.906933    |    0.078667    |    0.828267    |
--------------------------------------------------------------------------------------------------------------------------------
|        77aae641        |    0.820000    |    0.153333    |    0.666667    |    0.983767    |    0.014900    |    0.968867    |
--------------------------------------------------------------------------------------------------------------------------------
|        a296521b        |    0.913333    | -- 0.013333 -- |    0.900000    |    0.997333    | -- 0.000933 -- |    0.996400    |
--------------------------------------------------------------------------------------------------------------------------------
|        a4ff4885        |    0.786667    |    0.120000    |    0.666667    |    0.990467    |    0.002600    |    0.987867    |
--------------------------------------------------------------------------------------------------------------------------------
|        d7f43dca        |    0.810000    |    0.143333    |    0.666667    |    0.992233    |    0.006300    |    0.985933    |
--------------------------------------------------------------------------------------------------------------------------------
|        e167e7b3        |    0.920000    |    0.020000    |    0.900000    |    0.992200    |    0.003933    |    0.988267    |
--------------------------------------------------------------------------------------------------------------------------------
|        f11882e5        |    0.813333    |    0.146667    |    0.666667    |    0.921233    |    0.077367    |    0.843867    |
================================================================================================================================

~~~  [ info ] Best Parameters
----------------------------------------------------------------------------------------------------
acc  (00ad4e44) (0.943333 ± 0.030000)
----------------------------------------------------------------------------------------------------
{'optimizer': 'rmsprop', 'optimizer_config': {'lr': 0.022916225151133666}}
----------------------------------------------------------------------------------------------------
auc  (00ad4e44) (0.997467 ± 0.001067)
----------------------------------------------------------------------------------------------------
{'optimizer': 'rmsprop', 'optimizer_config': {'lr': 0.022916225151133666}}
----------------------------------------------------------------------------------------------------
best (00ad4e44)
----------------------------------------------------------------------------------------------------
{'optimizer': 'rmsprop', 'optimizer_config': {'lr': 0.022916225151133666}}
----------------------------------------------------------------------------------------------------

~~  [ info ] Results
================================================================================================================================
|        metrics         |                       acc                        |                       auc                        |
--------------------------------------------------------------------------------------------------------------------------------
|                        |      mean      |      std       |     score      |      mean      |      std       |     score      |
--------------------------------------------------------------------------------------------------------------------------------
|          fcnn          |    0.953333    |    0.000000    |    0.953333    |    0.998800    |    0.000000    |    0.998800    |
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

It is also worth mention that we can pass file datasets into `cflearn.tune_with` as well. See `tests/basic_usages.py` for more details.


## License

`carefree-learn` is MIT licensed, as found in the [LICENSE](https://github.com/carefree0910/carefree-learn/blob/master/LICENSE) file.
