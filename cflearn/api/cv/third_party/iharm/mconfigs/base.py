from ..model.base import ISEUNetV1
from ..model.base import DeepImageHarmonization
from ..model.base import SSAMImageHarmonization


BMCONFIGS = {
    "dih256": {"model": DeepImageHarmonization, "params": {"depth": 7}},
    "improved_dih256": {
        "model": DeepImageHarmonization,
        "params": {"depth": 7, "batchnorm_from": 2, "image_fusion": True},
    },
    "improved_sedih256": {
        "model": DeepImageHarmonization,
        "params": {
            "depth": 7,
            "batchnorm_from": 2,
            "image_fusion": True,
            "attend_from": 5,
        },
    },
    "ssam256": {
        "model": SSAMImageHarmonization,
        "params": {"depth": 4, "batchnorm_from": 2, "attend_from": 2},
    },
    "improved_ssam256": {
        "model": SSAMImageHarmonization,
        "params": {
            "depth": 4,
            "ch": 32,
            "image_fusion": True,
            "attention_mid_k": 0.5,
            "batchnorm_from": 2,
            "attend_from": 2,
        },
    },
    "iseunetv1_256": {
        "model": ISEUNetV1,
        "params": {"depth": 4, "batchnorm_from": 2, "attend_from": 1, "ch": 64},
    },
    "dih512": {"model": DeepImageHarmonization, "params": {"depth": 8}},
    "improved_dih512": {
        "model": DeepImageHarmonization,
        "params": {"depth": 8, "batchnorm_from": 2, "image_fusion": True},
    },
    "improved_ssam512": {
        "model": SSAMImageHarmonization,
        "params": {
            "depth": 6,
            "ch": 32,
            "image_fusion": True,
            "attention_mid_k": 0.5,
            "batchnorm_from": 2,
            "attend_from": 3,
        },
    },
    "improved_sedih512": {
        "model": DeepImageHarmonization,
        "params": {
            "depth": 8,
            "batchnorm_from": 2,
            "image_fusion": True,
            "attend_from": 6,
        },
    },
}
