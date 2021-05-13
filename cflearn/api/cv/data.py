import cflearn

from torch import Tensor
from typing import Tuple
from typing import Union
from typing import Callable
from typing import Optional
from cflearn.types import tensor_dict_type
from cflearn.constants import INPUT_KEY
from cflearn.constants import LABEL_KEY
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torch.utils.data import Dataset
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
            actual_labels = labels.view(-1, 1)
        else:
            actual_labels = label_callback(batch)
        return {
            cflearn.INPUT_KEY: img,
            cflearn.LABEL_KEY: actual_labels,
            cflearn.ORIGINAL_LABEL_KEY: labels,
        }

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


class TensorDataset(Dataset):
    def __init__(self, x: Tensor, y: Optional[Tensor]):
        self.x = x
        self.y = y

    def __getitem__(self, index: int) -> tensor_dict_type:
        return {
            INPUT_KEY: self.x[index],
            LABEL_KEY: 0 if self.y is None else self.y[index],
        }

    def __len__(self) -> int:
        return self.x.shape[0]


def get_tensor_loader(
    x: Tensor,
    y: Optional[Tensor],
    *,
    shuffle: bool = True,
    batch_size: int = 64,
) -> DLLoader:
    data = DLData(TensorDataset(x, y))
    return DLLoader(DataLoader(data, batch_size, shuffle))  # type: ignore


def get_tensor_loaders(
    x_train: Tensor,
    y_train: Optional[Tensor] = None,
    x_valid: Optional[Tensor] = None,
    y_valid: Optional[Tensor] = None,
) -> Tuple[DLLoader, Optional[DLLoader]]:
    train_loader = get_tensor_loader(x_train, y_train)
    if x_valid is None:
        return train_loader, None
    valid_loader = get_tensor_loader(x_valid, y_valid)
    return train_loader, valid_loader
