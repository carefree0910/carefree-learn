from ..base import ModelBase


@ModelBase.register("linear")
@ModelBase.register_pipe("linear")
class LinearModel(ModelBase):
    def define_pipe_configs(self) -> None:
        cfg = self.get_core_config(self)
        cfg["linear_config"] = self.config.setdefault("linear_config", {})
        self.define_head_config("linear", cfg)


__all__ = ["LinearModel"]
