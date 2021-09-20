# type: ignore

import os
import cflearn

from PIL import Image
from cflearn.misc.toolkit import check_is_ci
from cflearn.misc.toolkit import download_dataset
from cflearn.models.cv.stylizer.constants import STYLE_KEY


is_ci = check_is_ci()
Image.MAX_IMAGE_PIXELS = None

data_root = "data"
dataset = "adain_tiny"
hierarchy = [data_root]
if is_ci:
    hierarchy.append(dataset)
content_folder = os.path.join(*hierarchy, "contents")
style_folder = os.path.join(*hierarchy, "styles")
gathered_folder = os.path.join(*hierarchy, "gathered")
lmdb_config: dict = {}


if __name__ == "__main__":
    if is_ci and not os.path.isdir(style_folder):
        download_dataset(dataset, root=data_root)
    cflearn.cv.prepare_image_folder(
        content_folder,
        gathered_folder,
        to_index=False,
        label_fn=None,
        num_jobs=0 if is_ci else 24,
        lmdb_config=lmdb_config,
    )
    data = cflearn.cv.StyleTransferData(
        gathered_folder,
        style_folder,
        batch_size=8,
        num_workers=2 if is_ci else 4,
        transform=cflearn.cv.StyleTransferTransform(label_alias=STYLE_KEY),
        test_transform=cflearn.cv.StyleTransferTestTransform(label_alias=STYLE_KEY),
        lmdb_config=lmdb_config,
    )

    m = cflearn.DLZoo.load_pipeline(
        "style_transfer/adain",
        callback_names=["adain", "mlflow"],
        debug=is_ci,
    )
    m.fit(data, cuda=None if is_ci else 4)
