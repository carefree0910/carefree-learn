from setuptools import setup
from setuptools import find_packages


VERSION = "0.4.0"
DESCRIPTION = "Deep Learning with PyTorch made easy"
with open("README.md", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

cv_requires = [
    "ftfy",
    "lmdb",
    "regex",
    "transformers",
    "albumentations",
    "pillow",
    "scikit-image",
    "scipy>=1.8.0",
    "opencv-python-headless",
]
cv_full_requires = cv_requires + [
    "timm",
    "salesforce-lavis",
]
onnx_requires = [
    "onnx",
    "onnxruntime",
    "onnx-simplifier>=0.4.1",
]

setup(
    name="carefree-learn",
    version=VERSION,
    packages=find_packages(exclude=("tests",)),
    install_requires=[
        "accelerate",
        "safetensors",
        "carefree-toolkit>=0.3.4",
    ],
    extras_require={
        "onnx": onnx_requires,
        "cv": cv_requires,
        "cv_full": cv_full_requires,
        "full": cv_full_requires
        + onnx_requires
        + [
            "open_clip_torch",
            "faiss-cpu",
            "protobuf==3.19.4",
            "ortools>=9.3.0",
            "sacremoses",
            "sentencepiece",
            "mlflow",
            "plotly",
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
