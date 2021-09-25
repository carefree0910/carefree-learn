import os
import cflearn

from PIL import Image
from cflearn.constants import DATA_CACHE_DIR
from cflearn.misc.toolkit import check_is_ci
from cflearn.misc.toolkit import download_dataset


is_ci = check_is_ci()

data_folder = DATA_CACHE_DIR if is_ci else "data"
dataset = f"poster{'_tiny' if is_ci else ''}"
src_folder = os.path.join(data_folder, dataset)
tgt_folder = os.path.join(data_folder, "poster_data")

img_size = 224
lmdb_config: dict = {}
Image.MAX_IMAGE_PIXELS = None


if __name__ == "__main__":
    if is_ci and not os.path.isdir(src_folder):
        download_dataset(dataset, root=data_folder)
    data = cflearn.cv.prepare_image_folder_data(
        src_folder,
        tgt_folder,
        to_index=False,
        batch_size=4 if is_ci else 32,
        num_workers=0 if is_ci else 10,
        transform="ssl",
        transform_config={"img_size": img_size},
        test_transform="ssl_test",
        test_transform_config={"img_size": img_size},
        make_labels_in_parallel=False,
        num_jobs=0,
        lmdb_config=lmdb_config,
    ).data

    m = cflearn.api.dino(callback_names="mlflow", amp=not is_ci, debug=is_ci)
    m.fit(data, cuda=None if is_ci else 0)
