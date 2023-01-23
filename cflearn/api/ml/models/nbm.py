import os

import numpy as np

from typing import List
from typing import Tuple
from typing import Optional
from dataclasses import dataclass
from cftool.misc import print_warning
from cftool.array import to_numpy
from cftool.array import to_torch

from ..pipeline import MLPipeline
from ....misc.toolkit import eval_context
from ....models.ml.nbm import NBM
from ....models.schemas.ml import IEncoder
from ....models.schemas.ml import Dimensions

try:
    import matplotlib.pyplot as plt
except:
    plt = None


@dataclass
class NBMInspector:
    m: NBM
    encoder: Optional[IEncoder] = None
    dimensions: Optional[Dimensions] = None
    dpi: int = 200
    figsize: Tuple[int, int] = (8, 6)
    boundings: Optional[List[Tuple[float, float]]] = None
    column_names: Optional[List[str]] = None

    def fit(
        self,
        x: np.ndarray,
        column_names: Optional[List[str]] = None,
    ) -> "NBMInspector":
        self.boundings = []
        for i in range(x.shape[-1]):
            column = x[..., i]
            self.boundings.append((column.min().item(), column.max().item()))
        self.column_names = column_names
        return self

    def visualize(
        self,
        target_x_dims: List[Tuple[int, ...]],
        y_dim: int,
        export_folder: str,
        *,
        dense: int = 400,
        device: str = "cpu",
    ) -> None:
        if plt is None:
            raise ValueError("`matplotlib` is needed for `visualize`")
        if self.boundings is None:
            msg = "`boundings` is not defined, did you forget to call `fit`?"
            raise ValueError(msg)
        os.makedirs(export_folder, exist_ok=True)
        self.m.to(device)
        existing_titles = set()
        for x_dims in target_x_dims:
            if len(x_dims) != 1:
                print_warning("dim > 1 is not supported yet")
                continue

            if self.column_names is None:
                title = "-".join(str(i) for i in sorted(x_dims))
            else:
                title = "-".join(self.column_names[i] for i in sorted(x_dims))
            title = f"{title} ({y_dim})"
            if title in existing_titles:
                continue
            existing_titles.add(title)

            # currently only support one dim
            dim = x_dims[0]
            if self.dimensions is None:
                indices = x_dims
                is_categorical = False
            else:
                res = self.dimensions.get_indices_in_merged(dim)
                if res is None:
                    print_warning(f"column {dim} is redundant, skipped")
                    continue
                indices, is_categorical = res
            i_min, i_max = self.boundings[dim]
            if not is_categorical:
                x = np.random.random([dense, 1]) * (i_max - i_min) + i_min
                x = x[np.argsort(x.ravel())]
                net = to_torch(x).to(device)
                net = self.m.inspect(net, indices, y_dim, already_extracted=True)
                response = to_numpy(net)
                self._export_numerical(
                    x,
                    response,
                    title,
                    os.path.join(export_folder, f"{title}.png"),
                )
            else:
                if self.encoder is None:
                    print(
                        f"column {dim} is categorical "
                        "but `encoder` is not provided, skipped"
                    )
                    continue
                x = np.arange(i_min, i_max + 1)[..., None]
                with eval_context(self.encoder):
                    encoded = self.encoder.encode(x, dim)
                net = encoded.merged
                response = np.zeros([net.shape[0], 1], np.float32)
                for i, idx in enumerate(indices):
                    i_net = net[..., [i]]
                    i_pack = (idx,)
                    i_net = self.m.inspect(i_net, i_pack, y_dim, already_extracted=True)
                    response += to_numpy(i_net)
                self._export_categorical(
                    len(x),
                    response,
                    title,
                    os.path.join(export_folder, f"{title}.png"),
                )

    def _export_numerical(
        self,
        x: np.ndarray,
        response: np.ndarray,
        title: str,
        export_path: str,
    ) -> None:
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        plt.title(title)
        plt.plot(x.ravel(), response.ravel(), label="response")
        plt.legend()
        fig.savefig(export_path)
        plt.close()

    def _export_categorical(
        self,
        n: int,
        response: np.ndarray,
        title: str,
        export_path: str,
    ) -> None:
        x_base = np.arange(1, n + 1)
        response = response.ravel()
        y_min, y_max = response.min(), response.max()
        padding = 0.1 * (y_max - y_min)
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        plt.title(title)
        plt.bar(
            x_base,
            response,
            width=0.35,
            edgecolor="white",
        )
        plt.xticks(
            [i for i in range(n + 2)],
            [""] + [str(i) for i in range(n)] + [""],
        )
        plt.setp(plt.xticks()[1], rotation=30, horizontalalignment="right")
        plt.ylim(y_min - padding, y_max + padding)
        fig.tight_layout()
        fig.savefig(export_path)
        plt.close()

    @classmethod
    def from_pipeline(
        cls,
        m: MLPipeline,
        dpi: int = 200,
        figsize: Tuple[int, int] = (8, 6),
    ) -> "NBMInspector":
        model = m.model
        if isinstance(model.dimensions, list):
            raise ValueError(
                "`MLPipeline` with `_num_repeat` defined is detected, "
                "which is not compatible with `NBM.from_pipeline`"
            )
        return cls(
            model.core.core,
            model.encoder,
            model.dimensions,
            dpi,
            figsize,
        )


__all__ = [
    "NBMInspector",
]
