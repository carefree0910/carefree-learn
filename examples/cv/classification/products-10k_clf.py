# type: ignore

import os
import cflearn

from typing import List
from cflearn.constants import DATA_CACHE_DIR
from cflearn.misc.toolkit import check_is_ci
from cflearn.misc.toolkit import download_dataset


is_ci = check_is_ci()

data_folder = DATA_CACHE_DIR if is_ci else "data"
dataset = f"products-10k_clf{'_tiny' if is_ci else ''}"
src_folder = os.path.join(data_folder, dataset)
tgt_folder = os.path.join(data_folder, "products-10k_data")


class Products10kPreparation(cflearn.DefaultPreparation):
    def get_label(self, hierarchy: List[str]) -> int:
        return int(hierarchy[2] == "pos")


if __name__ == "__main__":
    if is_ci and not os.path.isdir(src_folder):
        download_dataset(dataset, root=data_folder)
    data = cflearn.prepare_image_folder_data(
        src_folder,
        tgt_folder,
        to_index=False,
        batch_size=16,
        preparation=Products10kPreparation(),
        num_workers=4,
        drop_train_last=not is_ci,
        transform=["a_resize", "to_tensor"],
        num_jobs=0 if is_ci else 8,
    ).data

    m = cflearn.api.resnet18(
        2,
        False,
        model_config={"in_channels": 4},
        metric_names="acc" if is_ci else ["acc", "auc"],
        callback_names=["clf", "mlflow"],
        debug=is_ci,
    )
    m.fit(data, cuda=None if is_ci else 7)
