# type: ignore

import cflearn
import argparse

from u2net_finetune import prepare
from u2net_finetune import pretrained_ckpt

# CI
parser = argparse.ArgumentParser()
parser.add_argument("--ci", type=int, default=0)
args = parser.parse_args()
is_ci = bool(args.ci)


finetune_ckpt = "pretrained/lite_finetune_aug.pt"

if __name__ == "__main__":
    train, valid = cflearn.cv.get_image_folder_loaders(
        prepare(is_ci),
        batch_size=16,
        num_workers=2 if is_ci else 4,
        transform=cflearn.cv.ABundle(label_alias="mask"),
        test_transform=cflearn.cv.ABundleTest(label_alias="mask"),
    )
    m = cflearn.cv.CarefreePipeline(
        "cascade_u2net",
        {
            "in_channels": 3,
            "out_channels": 1,
            "lv1_model_ckpt_path": pretrained_ckpt if is_ci else finetune_ckpt,
            "lv2_model_config": {"lite": True},
            "lite": True,
        },
        loss_name="sigmoid_mae",
        callback_names=["unet", "mlflow"],
        callback_configs={"mlflow": {"experiment_name": "lite_refine"}},
        scheduler_name="none",
        fixed_steps=1 if is_ci else None,
    )
    m.fit(train, valid, cuda=None if is_ci else 5)
    # m.ddp(train, valid, cuda_list=[4, 5])
