import cflearn

from torch import Tensor
from typing import Tuple
from typing import Callable
from typing import Optional
from cflearn.types import tensor_dict_type
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from cflearn.misc.internal_ import DLData
from cflearn.misc.internal_ import DLLoader


def get_mnist(
    *,
    batch_size: int = 64,
    transform: Optional[Callable] = None,
    label_callback: Optional[Callable[[Tuple[Tensor, Tensor]], Tensor]] = None,
) -> Tuple[DLLoader, DLLoader]:
    def batch_callback(batch: Tuple[Tensor, Tensor]) -> tensor_dict_type:
        img, labels = batch
        if label_callback is None:
            labels = labels.view(-1, 1)
        else:
            labels = label_callback(batch)
        return {cflearn.INPUT_KEY: img, cflearn.LABEL_KEY: labels}

    train_data = DLData(MNIST("data", transform=transform, download=True))
    valid_data = DLData(MNIST("data", train=False, transform=transform, download=True))

    train_pt_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)  # type: ignore
    valid_pt_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)  # type: ignore

    train_loader = DLLoader(train_pt_loader, batch_callback)
    valid_loader = DLLoader(valid_pt_loader, batch_callback)
    return train_loader, valid_loader
