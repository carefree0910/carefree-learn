# type: ignore

import cflearn

from u2net_finetune import prepare
from cflearn.misc.toolkit import check_is_ci
from cflearn.misc.toolkit import download_model


is_ci = check_is_ci()
finetune_ckpt = "path/to/your/finetune/model"
if is_ci:
    finetune_ckpt = download_model("u2net.lite")

if __name__ == "__main__":
    data = cflearn.cv.ImageFolderData(
        prepare(is_ci),
        batch_size=16,
        num_workers=2 if is_ci else 4,
        transform=cflearn.cv.ABundleTransform(label_alias="mask"),
        test_transform=cflearn.cv.ABundleTestTransform(label_alias="mask"),
    )

    m = cflearn.api.u2net_lite_refine(
        finetune_ckpt,
        callback_names=["cascade_u2net", "mlflow"],
        debug=is_ci,
    )
    m.fit(data, cuda=None if is_ci else 5)
