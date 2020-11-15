from .base import ModelBase


@ModelBase.register("tree_dnn")
@ModelBase.register_pipe("dndf")
@ModelBase.register_pipe("fcnn", transform="embedding", head_config="pruned")
class TreeDNN(ModelBase):
    pass


@ModelBase.register("tree_stack")
@ModelBase.register_pipe("tree_stack")
class TreeStack(ModelBase):
    @property
    def output_probabilities(self) -> bool:
        return True


@TreeStack.register("tree_linear")
@ModelBase.register_pipe("tree_stack", head_config="linear")
class TreeLinear(TreeStack):
    pass


__all__ = ["TreeDNN", "TreeStack", "TreeLinear"]
