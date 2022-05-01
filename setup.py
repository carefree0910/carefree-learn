from setuptools import setup, find_packages

VERSION = "0.2.1"
DESCRIPTION = "Deep Learning with PyTorch made easy"
with open("README.md", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

setup(
    name="carefree-learn",
    version=VERSION,
    packages=find_packages(exclude=("tests",)),
    install_requires=[
        "ftfy",
        "regex",
        "lmdb",
        "einops",
        "albumentations",
        "scikit-image",
        "numba>=0.50",
        "mlflow",
        "onnx",
        "onnx-simplifier",
        "onnxruntime",
        "plotly",
        "carefree-data>=0.2.6",
        "carefree-toolkit>=0.2.9",
        "dill",
        "future",
        "psutil",
        "cython>=0.29.28",
        "numpy>=1.22.3",
        "scipy>=1.8.0",
        "scikit-learn>=1.0.2",
        "matplotlib>=3.5.1",
    ],
    author="carefree0910",
    author_email="syameimaru.saki@gmail.com",
    url="https://github.com/carefree0910/carefree-learn",
    download_url=f"https://github.com/carefree0910/carefree-learn/archive/v{VERSION}.tar.gz",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="python machine-learning deep-learning solution PyTorch",
)
