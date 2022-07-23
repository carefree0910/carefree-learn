from setuptools import setup
from setuptools import find_packages


VERSION = "0.3.0"
DESCRIPTION = "Deep Learning with PyTorch made easy"
with open("README.md", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

setup(
    name="carefree-learn",
    version=VERSION,
    packages=find_packages(exclude=("tests",)),
    install_requires=["carefree-toolkit>=0.2.12"],
    extras_require={
        "onnx": [
            "onnx",
            "onnx-simplifier",
            "onnxruntime",
        ],
        "full": [
            "protobuf==3.19.4",
            "ortools>=9.3.0",
            "sacremoses",
            "sentencepiece",
            "transformers",
            "ftfy",
            "regex",
            "lmdb",
            "albumentations",
            "mlflow",
            "onnx",
            "onnx-simplifier",
            "onnxruntime",
            "plotly",
            "carefree-data>=0.2.8",
            "carefree-ml>=0.1.3",
            "carefree-cv>=0.1.0",
        ],
    },
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
