import numpy as np

from typing import Any
from typing import Dict
from typing import Union
from cftool.misc import shallow_copy_dict
from sklearn.tree import DecisionTreeClassifier

from ..base import HeadConfigs
from ....misc.toolkit import to_torch


@HeadConfigs.register("nnb_meta", "default")
class DefaultNNBMetaConfig(HeadConfigs):
    def should_bypass(self, config: Dict[str, Any]) -> Union[bool, Dict[str, bool]]:
        mnb, normal = config["nnb_mnb"], config["nnb_normal"]
        return {
            "nnb_mnb": mnb["categorical"] is None,
            "nnb_normal": normal["numerical"] is None,
        }

    def get_default(self) -> Dict[str, Any]:
        # prepare
        x, y = self.tr_data.processed.xy
        split = self.dimensions.split_features(to_torch(x), np.arange(len(x)), "tr")
        if not self.dimensions.has_numerical:
            numerical = None
        else:
            numerical = split.numerical
        if self.dimensions.one_hot_dim == 0:
            categorical = None
        else:
            assert split.categorical is not None
            categorical = split.categorical.one_hot
        common_config = {"y_ravel": y.ravel()}
        # mnb
        mnb_config = shallow_copy_dict(common_config)
        mnb_config.update(
            {
                "categorical": categorical,
                "categorical_dims": self.dimensions.categorical_dims,
            }
        )
        # normal
        normal_config = shallow_copy_dict(common_config)
        normal_config.update({"numerical": numerical, "pretrain": True})
        return {"nnb_mnb": mnb_config, "nnb_normal": normal_config}


@HeadConfigs.register("ndt", "default")
class DefaultNDTConfig(HeadConfigs):
    def get_default(self) -> Dict[str, Any]:
        # decision tree
        x, y = self.tr_data.processed.xy
        msg = "fitting decision tree"
        self.log_msg(msg, self.info_prefix, verbose_level=2)  # type: ignore
        split = self.dimensions.split_features(to_torch(x), np.arange(len(x)), "tr")
        x_merge = split.merge().cpu().numpy()
        dt = DecisionTreeClassifier(max_depth=10, random_state=142857)
        dt.fit(x_merge, y.ravel(), sample_weight=self.tr_weights)
        return {
            "dt": dt,
            "activations": {"planes": "sign", "routes": "multiplied_softmax"},
            "activation_configs": {},
        }


__all__ = [
    "DefaultNNBMetaConfig",
    "DefaultNDTConfig",
]
