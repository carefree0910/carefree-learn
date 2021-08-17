import os
import urllib.request

from tqdm import tqdm
from typing import Optional
from zipfile import ZipFile

from ..constants import WARNING_PREFIX


class DownloadProgressBar(tqdm):
    def update_to(
        self,
        b: int = 1,
        bsize: int = 1,
        tsize: Optional[int] = None,
    ) -> None:
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_dataset(
    name: str,
    *,
    root: str = os.getcwd(),
    remove_zip: Optional[bool] = None,
    extract_zip: bool = True,
    prefix: str = "https://github.com/carefree0910/datasets/releases/download/latest/",
) -> None:
    os.makedirs(root, exist_ok=True)
    file = f"{name}.zip"
    tgt_zip_path = os.path.join(root, file)
    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=name) as t:
        urllib.request.urlretrieve(
            f"{prefix}{file}",
            filename=tgt_zip_path,
            reporthook=t.update_to,
        )
    if extract_zip:
        with ZipFile(tgt_zip_path, "r") as zip_ref:
            zip_ref.extractall(os.path.join(root, name))
    if remove_zip is None:
        remove_zip = extract_zip
    if remove_zip:
        if extract_zip:
            os.remove(tgt_zip_path)
        else:
            print(f"{WARNING_PREFIX}zip file is not extracted, so we'll not remove it!")


__all__ = [
    "download_dataset",
]
