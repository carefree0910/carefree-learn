# type: ignore

import cflearn

from cflearn.misc import U2NetCallback

from u2net_finetune import prepare


src_folder = "data/raw"
src_rgba_folder = "data/rgba"
tgt_folder = "data/products-10k"
label_folder = "data/products-10k_labels"


@U2NetCallback.register("cascade_u2net")
class CascadeU2NetCallback(U2NetCallback):
    pass


if __name__ == "__main__":
    prepare()
    train_loader, valid_loader = cflearn.cv.get_image_folder_loaders(
        tgt_folder,
        batch_size=16,
        num_workers=4,
        transform="for_salient_object_detection",
        test_transform="for_salient_object_detection_test",
    )
    m = cflearn.cv.CarefreePipeline(
        "cascade_u2net",
        {
            "in_channels": 3,
            "out_channels": 1,
            "lv1_model_ckpt_path": "pretrained/model_200343_all.pt",
            "lv2_resolution": 512,
            "lite": False,
        },
        loss_name="sigmoid_mae",
        callback_names=["unet", "mlflow"],
        callback_configs={"mlflow": {"experiment_name": "large_refine"}},
        lr=1.0e-4,
    )
    m.fit(train_loader, valid_loader, cuda="7")
