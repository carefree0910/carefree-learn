import cflearn
import unittest

from typing import Any
from typing import Dict

try:
    import transformers
except ImportError:
    transformers = None
try:
    import cfcv
except ImportError:
    cfcv = None


def _inject_mock(req: Dict[str, Any], d: Dict[str, Any]) -> None:
    for k, v in req.items():
        kd = d.setdefault(k, {})
        if isinstance(v, dict):
            _inject_mock(v, kd)
            continue
        assert isinstance(v, list)
        for vv in v:
            if vv == "img_size":
                kd[vv] = 112
            elif vv == "num_classes":
                kd[vv] = 12
            elif vv == "pretrained_ckpt" or vv == "lv1_model_ckpt_path":
                kd[vv] = "foo.pt"
            # hugging_face/general
            elif vv == "model":
                kd[vv] = "peterchou/simbert-chinese-base"
            # hugging_face/opus
            elif vv == "src":
                kd[vv] = "zh"
            elif vv == "tgt":
                kd[vv] = "en"
            # clip_vqgan_aligner
            elif vv == "text":
                kd[vv] = "carefree-learn"


class TestZoo(unittest.TestCase):
    def test_model_zoo(self) -> None:
        models = cflearn.api.model_zoo(verbose=True)
        for model in models:
            if model.name.startswith("hugging_face") and transformers is None:
                continue
            # avoid OOM
            if model.name.startswith("diffusion"):
                continue
            kwargs: Dict[str, Any] = {}
            _inject_mock(model.requirements, kwargs)
            if model.name.startswith("vae/vanilla"):
                kwargs["model_config"]["latent_resolution"] = 16
            if "clip_vqgan_aligner" in model.name:
                if cfcv is None:
                    continue
                # doesn't need to download pretrained models here
                kwargs["model_config"]["perceptor_pretrained_name"] = None
                kwargs["model_config"]["generator_pretrained_name"] = None
            m = cflearn.api.from_zoo(model.name, build=True, **kwargs)
            m2 = cflearn.api.from_json(m.to_json())
            m3 = cflearn.api.from_zoo(model.name, build=False, **kwargs)
            m4 = cflearn.api.from_json(m3.to_json())
            self.assertEqual(m, m2)
            self.assertEqual(m3, m4)


if __name__ == "__main__":
    unittest.main()
