from ..base import ModelBase


@ModelBase.register("linear")
@ModelBase.register_pipe("linear")
class LinearModel(ModelBase):
    def define_head_configs(self) -> None:
        cfg = self.get_core_config(self)
        linear_config = self.config.setdefault("linear_config", {})
        cfg["linear_config"] = linear_config
        self.define_head_config("linear", cfg)


__all__ = ["LinearModel"]
