# type: ignore

import cflearn

from u2net_finetune import prepare
from u2net_finetune import pretrained_ckpt
from cflearn.misc.toolkit import check_is_ci


is_ci = check_is_ci()

finetune_ckpt = "pretrained/lite_finetune_aug.pt"

if __name__ == "__main__":
    data = cflearn.cv.ImageFolderData(
        prepare(is_ci),
        batch_size=16,
        num_workers=2 if is_ci else 4,
        transform=cflearn.cv.ABundleTransform(label_alias="mask"),
        test_transform=cflearn.cv.ABundleTestTransform(label_alias="mask"),
    )

    m = cflearn.DLZoo.load_pipeline(
        "segmentor/u2net.refine_lite",
        lv1_model_ckpt_path=pretrained_ckpt if is_ci else finetune_ckpt,
        callback_names=["cascade_u2net", "mlflow"],
        debug=is_ci,
    )
    m.fit(data, cuda=None if is_ci else 5)
