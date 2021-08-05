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
        batch_size=4,
        num_workers=2,
        transform="for_salient_object_detection",
        test_transform="for_salient_object_detection_test",
    )
    cflearn.MultiStageLoss.register_(["bce", "iou", "sigmoid_mae"])
    m = cflearn.cv.CarefreePipeline(
        "cascade_u2net",
        {
            "in_channels": 3,
            "out_channels": 1,
            "lv1_model_ckpt_path": "pretrained/model_3269_bce.pt",
            "lv2_resolution": 512,
            "lv2_model_config": {"lite": True},
            "lite": False,
        },
        loss_name="multi_stage_bce_iou_sigmoid_mae",
        loss_metrics_weights={"bce0": 0.2, "iou0": 0.4, "sigmoid_mae0": 0.4},
        scheduler_name="none",
    )
    m.fit(train_loader, valid_loader, cuda="7")
