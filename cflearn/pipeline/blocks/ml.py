from typing import Union
from typing import Optional
from cftool.misc import print_warning
from cftool.misc import shallow_copy_dict

from .basic import SetDefaultsBlock
from .basic import SetTrainerDefaultsBlock
from ..core import Block
from ...schema import DLConfig
from ...schema import MLConfig
from ...schema import MLEncoderSettings
from ...data.ml import MLData
from ...data.blocks import ColumnTypes
from ...data.blocks import RecognizerBlock


# static blocks


@Block.register("set_ml_defaults")
class SetMLDefaultsBlock(SetDefaultsBlock):
    data: MLData

    def build(self, config: Union[DLConfig, MLConfig]) -> None:
        super().build(config)
        if not isinstance(config, MLConfig):
            return
        self._infer_encoder_settings(config)
        b_recognizer = self.b_recognizer
        if b_recognizer is not None:
            config.index_mapping = shallow_copy_dict(b_recognizer.index_mapping)

    @property
    def b_recognizer(self) -> Optional[RecognizerBlock]:
        return self.data.processor.try_get_block(RecognizerBlock)

    def _infer_encoder_settings(self, config: MLConfig) -> None:
        if config.encoder_settings is not None:
            return
        if not config.infer_encoder_settings:
            return
        if self.data is None:
            print_warning(
                "`infer_encoder_settings` is set to True "
                "but `data` is not provided, it will take no effect."
            )
            return
        b_recognizer = self.b_recognizer
        if b_recognizer is None:
            print_warning(
                "`infer_encoder_settings` is set to True "
                "but `RecognizerBlock` is not provided, it will take no effect"
            )
            return
        encoder_settings = {}
        for original_idx in b_recognizer.index_mapping:
            if b_recognizer.feature_types[original_idx] == ColumnTypes.CATEGORICAL:
                dim = b_recognizer.num_unique_features[original_idx]
                encoder_settings[original_idx] = MLEncoderSettings(dim)
        config.encoder_settings = encoder_settings
        d = {k: v.asdict() for k, v in encoder_settings.items()}
        self._defaults["encoder_settings"] = d


@Block.register("set_ml_trainer_defaults")
class SetMLTrainerDefaultsBlock(SetTrainerDefaultsBlock):
    def build(self, config: DLConfig) -> None:
        if config.monitor_names is None:
            config.monitor_names = ["mean_std", "plateau"]
            self._defaults["monitor_names"] = ["mean_std", "plateau"]
        super().build(config)


__all__ = [
    "SetMLDefaultsBlock",
    "SetMLTrainerDefaultsBlock",
]
