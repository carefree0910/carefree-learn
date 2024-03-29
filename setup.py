from setuptools import setup
from setuptools import find_packages


VERSION = "0.5.0"
DESCRIPTION = "Deep Learning with PyTorch made easy"
with open("README.md", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

onnx_requires = [
    "onnx",
    "onnxruntime",
    "onnx-simplifier>=0.4.1",
]
ml_requires = [
    "captum",
    "mlflow",
]
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
cv_full_requires = (
    cv_requires
    + onnx_requires
    + [
        "timm",
        "salesforce-lavis",
        "xformers>=0.0.19",
    ]
)

setup(
    name="carefree-learn",
    version=VERSION,
    packages=find_packages(exclude=("tests",)),
    install_requires=[
        "filelock",
        "accelerate",
        "safetensors",
        "carefree-toolkit>=0.3.12",
    ],
    extras_require={
        "onnx": onnx_requires,
        "ml": ml_requires,
        "ml_full": ml_requires + onnx_requires,
        "cv": cv_requires,
        "cv_full": cv_full_requires,
        "full": ml_requires
        + cv_full_requires
        + [
            "open_clip_torch",
            "faiss-cpu",
            "protobuf==3.19.4",
            "ortools>=9.3.0",
            "sacremoses",
            "sentencepiece",
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
