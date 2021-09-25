# type: ignore

import os
import cv2
import cflearn

import numpy as np

from typing import List
from cflearn.constants import DATA_CACHE_DIR
from cflearn.misc.toolkit import check_is_ci
from cflearn.misc.toolkit import download_dataset
from cflearn.misc.toolkit import min_max_normalize
from cflearn.api.cv.data.interface import ImageFolderData


is_ci = check_is_ci()


class U2NETPreparation(cflearn.cv.DefaultPreparation):
    def __init__(self, src_rgba_folder: str, label_folder: str):
        self.rgba_folder = src_rgba_folder
        self.label_folder = label_folder

    def get_label(self, hierarchy: List[str]) -> str:
        file_id = os.path.splitext(hierarchy[-1])[0]
        os.makedirs(self.label_folder, exist_ok=True)
        label_path = os.path.abspath(os.path.join(self.label_folder, f"{file_id}.npy"))
        if os.path.isfile(label_path):
            return label_path
        rgba_path = os.path.abspath(os.path.join(self.rgba_folder, f"{file_id}.png"))
        alpha = cv2.imread(rgba_path, cv2.IMREAD_UNCHANGED)[..., -1:]
        alpha = min_max_normalize(alpha.astype(np.float32), global_norm=True)
        np.save(label_path, alpha)
        return label_path


def get_data(ci: bool) -> ImageFolderData:
    data_root = DATA_CACHE_DIR if ci else "data"
    dataset = "products-10k_tiny"
    if not ci:
        data_folder = data_root
    else:
        data_folder = os.path.join(data_root, dataset)
    src_folder = os.path.join(data_folder, "raw")
    src_rgba_folder = os.path.join(data_folder, "rgba")
    tgt_folder = os.path.join(data_folder, "products-10k")
    label_folder = os.path.join(data_folder, "products-10k_labels")
    if ci and not os.path.isdir(src_folder):
        download_dataset(dataset, root=data_root)
    return cflearn.cv.prepare_image_folder_data(
        src_folder,
        tgt_folder,
        to_index=False,
        preparation=U2NETPreparation(src_rgba_folder, label_folder),
        batch_size=16,
        num_workers=2 if ci else 4,
        transform="a_bundle",
        transform_config={"label_alias": "mask"},
        test_transform="a_bundle_test",
        make_labels_in_parallel=not ci,
        num_jobs=0 if ci else 8,
    ).data


if __name__ == "__main__":
    data = get_data(is_ci)
    m = cflearn.api.u2net_lite_finetune(callback_names=["u2net", "mlflow"], debug=is_ci)
    m.fit(data, cuda=None if is_ci else 0)
