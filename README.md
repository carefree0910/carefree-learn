![carefree-learn][socialify-image]

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://demo.ailab.nolibox.com/)

Deep Learning with [PyTorch](https://pytorch.org/) made easy 🚀 !


## Carefree?

`carefree-learn` aims to provide **CAREFREE** usages for both users and developers. It also provides a [corresponding repo](https://github.com/carefree0910/carefree-learn-deploy) for production.

### User Side

#### Machine Learning 📈

```python
import cflearn
import numpy as np

x = np.random.random([1000, 10])
y = np.random.random([1000, 1])
config = cflearn.DLConfig(model_name="fcnn", model_config=dict(input_dim=10, output_dim=1), loss_name="mae")
m = cflearn.api.fit_ml(x, y, config=config)
```

#### Computer Vision 🖼️

```python
import cflearn

data = cflearn.mnist_data(additional_blocks=[cflearn.FlattenBlock()])
config = cflearn.DLConfig(
    model_name="fcnn",
    model_config=dict(input_dim=784, output_dim=10),
    loss_name="focal",
    metric_names=["acc", "auc"],
)
m = cflearn.DLTrainingPipeline.init(config).fit(data)
```

### Developer Side

> This is a WIP section :D

### Production Side

`carefree-learn` could be deployed easily because
+ It could be exported to `onnx` format with one line of code (`m.to_onnx(...)`)
+ A native repo called [`carefree-learn-deploy`](https://github.com/carefree0910/carefree-learn-deploy) could do the rest of the jobs, which uses `FastAPI`, `uvicorn` and `docker` as its backend.

> Please refer to [Quick Start](https://carefree0910.me/carefree-learn-doc/docs/getting-started/quick-start) and [Developer Guides](https://carefree0910.me/carefree-learn-doc/docs/developer-guides/general-customization) for detailed information.


## Why carefree-learn?

`carefree-learn` is a general Deep Learning framework based on PyTorch. Since `v0.2.x`, `carefree-learn` has extended its usage from tabular dataset to (almost) all kinds of dataset. In the mean time, the APIs remain (almost) the same as `v0.1.x`: still simple, powerful and easy to use!

Here are some main advantages that `carefree-learn` holds:

### Machine Learning 📈

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

### Computer Vision 🖼️

+ Also provides a [scikit-learn](https://scikit-learn.org/stable/)-like interface with much more 'carefree' usages.
+ Provides many out-of-the-box pre-trained models and well hand-crafted training defaults for reproduction & finetuning.
+ Seamlessly supports efficient `ddp` (simply call `cflearn.api.run_ddp("run.py")`, where `run.py` is your normal training script).
+ Bunch of utility functions for research and production.


## Installation

`carefree-learn` requires Python 3.8 / 3.9.

### Pre-Installing PyTorch

`carefree-learn` requires `pytorch>=1.12.0`. Please refer to [PyTorch](https://pytorch.org/get-started/locally/), and it is highly recommended to pre-install PyTorch with conda.

### pip installation

After installing PyTorch, installation of `carefree-learn` would be rather easy:

> If you pre-installed PyTorch with conda, remember to activate the corresponding environment!

```bash
pip install carefree-learn
```

or install the full version of `carefree-learn`, with various useful functions and utilities:

```bash
pip install carefree-learn[full]
```


## Docker

### Prepare

`carefree-learn` has already been published on DockerHub, so it can be pulled directly:

```bash
docker pull carefree0910/carefree-learn:dev
```

or can be built locally:

```bash
docker build -t carefree0910/carefree-learn:dev .
```

> Notice that the image is built with the `full` version of `carefree-learn`.

### Run

```bash
docker run --rm -it --gpus all carefree0910/carefree-learn:dev
```


## Examples

+ [Iris](https://carefree0910.me/carefree-learn-doc/docs/examples/iris) – perhaps the best known database to be found in the pattern recognition literature.
+ [Titanic](https://carefree0910.me/carefree-learn-doc/docs/examples/titanic) – the best, first challenge for you to dive into ML competitions and familiarize yourself with how the Kaggle platform works.
+ [Operations](https://carefree0910.me/carefree-learn-doc/docs/examples/operations) - toy datasets for us to illustrate how to build your own models in `carefree-learn`.


## Citation

If you use `carefree-learn` in your research, we would greatly appreciate if you cite this library using this Bibtex:

```
@misc{carefree-learn,
  year={2020},
  author={Yujian He},
  title={carefree-learn, Deep Learning with PyTorch made easy},
  howpublished={\url{https://https://github.com/carefree0910/carefree-learn/}},
}
```


## License

`carefree-learn` is MIT licensed, as found in the [`LICENSE`](https://carefree0910.me/carefree-learn-doc/docs/about/license) file.


[socialify-image]: https://socialify.git.ci/carefree0910/carefree-learn/image?description=1&descriptionEditable=Deep%20Learning%20%E2%9D%A4%EF%B8%8F%20PyTorch&forks=1&issues=1&logo=https%3A%2F%2Fraw.githubusercontent.com%2Fcarefree0910%2Fcarefree-learn-doc%2Fmaster%2Fstatic%2Fimg%2Flogo.min.svg&pattern=Floating%20Cogs&stargazers=1