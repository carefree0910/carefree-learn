import cflearn

from torch import Tensor
from typing import Tuple
from typing import Union
from typing import Callable
from typing import Optional
from cflearn.types import tensor_dict_type
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from cflearn.misc.internal_ import DLData
from cflearn.misc.internal_ import DLLoader


def get_mnist(
    *,
    shuffle: bool = True,
    batch_size: int = 64,
    transform: Optional[Union[str, Callable]] = None,
    label_callback: Optional[Callable[[Tuple[Tensor, Tensor]], Tensor]] = None,
) -> Tuple[DLLoader, DLLoader]:
    def batch_callback(batch: Tuple[Tensor, Tensor]) -> tensor_dict_type:
        img, labels = batch
        if label_callback is None:
            labels = labels.view(-1, 1)
        else:
            labels = label_callback(batch)
        return {cflearn.INPUT_KEY: img, cflearn.LABEL_KEY: labels}

    if isinstance(transform, str):
        if transform == "for_classification":
            transform = transforms.ToTensor()
        elif transform == "for_generation":
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Lambda(lambda t: t * 2.0 - 1.0),
                ]
            )
        else:
            raise NotImplementedError(f"'{transform}' transform is not implemented")

    train_data = DLData(MNIST("data", transform=transform, download=True))
    valid_data = DLData(MNIST("data", train=False, transform=transform, download=True))

    train_pt_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)  # type: ignore
    valid_pt_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=shuffle)  # type: ignore

    train_loader = DLLoader(train_pt_loader, batch_callback)
    valid_loader = DLLoader(valid_pt_loader, batch_callback)
    return train_loader, valid_loader
