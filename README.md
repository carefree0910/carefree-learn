# carefree-learn

`carefree-learn` is a minimal Automatic Machine Learning (AutoML) solution for tabular datasets based on [PyTorch](https://pytorch.org/).

## Carefree?

`carefree-learn` aims to provide **CAREFREE** usages for both users and developers.

### User Side

```python
import cflearn
import numpy as np

x = np.random.random([1000, 10])
y = np.random.random([1000, 1])
m = cflearn.make().fit(x, y)
```

### Developer Side

```python
import cflearn
import numpy as np
from cflearn import ModelBase

@ModelBase.register("wnd_full")
@ModelBase.register_pipe("fcnn")
@ModelBase.register_pipe("linear")
class WideAndDeepFull(ModelBase):
    pass

x = np.random.random([1000, 10])
y = np.random.random([1000, 1])
m = cflearn.make("wnd_full").fit(x, y)
```

> Please refer to [Quick Start](https://carefree0910.me/carefree-learn-doc/docs/getting-started/quick-start) and [Build Your Own Models](https://carefree0910.me/carefree-learn-doc/docs/developer-guides/build-your-own-models) for detailed information.

## Why carefree-learn?

`carefree-learn`

+ Provides a [scikit-learn](https://scikit-learn.org/stable/)-like interface with much more 'carefree' usages, including:
    + Automatically deals with data pre-processing.
    + Automatically handles datasets saved in files (.txt, .csv).
    + Supports [Distributed Training](https://carefree0910.me/carefree-learn-doc/docs/user-guides/distributed#distributed-training), which means hyper-parameter tuning can be very efficient in `carefree-learn`.
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

## Citation

If you use `carefree-learn` in your research, we would greatly appreciate if you cite this library using this Bibtex:

```
@misc{carefree-learn,
  year={2020},
  author={Yujian He},
  title={carefree-learn, a minimal Automatic Machine Learning (AutoML) solution for tabular datasets based on PyTorch},
  howpublished={\url{https://https://github.com/carefree0910/carefree-learn/}},
}
```


## License

`carefree-learn` is MIT licensed, as found in the [`LICENSE`](https://carefree0910.me/carefree-learn-doc/docs/about/license) file.
