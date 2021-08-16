# type: ignore

import cflearn

from u2net_finetune import prepare


src_folder = "data/raw"
src_rgba_folder = "data/rgba"
tgt_folder = "data/products-10k"
label_folder = "data/products-10k_labels"


if __name__ == "__main__":
    prepare()
    train_loader, valid_loader = cflearn.cv.get_image_folder_loaders(
        tgt_folder,
        batch_size=8,
        num_workers=2,
        transform=cflearn.cv.ABundle(label_alias="mask"),
        test_transform=cflearn.cv.ABundleTest(label_alias="mask"),
    )
    m = cflearn.cv.CarefreePipeline(
        "cascade_u2net",
        {
            "in_channels": 3,
            "out_channels": 1,
            "lv1_model_ckpt_path": "pretrained/light_0.001.pt",
            "lv2_resolution": 512,
            "lv2_model_config": {"eca_kernel_size": 3},
            "lite": True,
        },
        loss_name="sigmoid_mae",
        callback_names=["unet", "mlflow"],
        callback_configs={"mlflow": {"experiment_name": "lite_refine"}},
        scheduler_name="plateau",
    )
    m.fit(train_loader, valid_loader, cuda="5")
