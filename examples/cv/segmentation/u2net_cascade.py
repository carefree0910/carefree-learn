# type: ignore

import cflearn

from cflearn.api.cv import AlphaSegmentationCallback

from u2net_finetune import prepare
from u2net_finetune import U2NetCallback


src_folder = "raw"
src_rgba_folder = "rgba"
tgt_folder = "u2net_finetune"
label_folder = "u2net_finetune_labels"


@AlphaSegmentationCallback.register("cascade_u2net")
class CascadeU2NetCallback(U2NetCallback):
    pass


if __name__ == "__main__":
    prepare()
    train_loader, valid_loader = cflearn.cv.get_image_folder_loaders(
        tgt_folder,
        batch_size=16,
        num_workers=4,
        transform="for_salient_object_detection",
    )
    cflearn.MultiStageLoss.register_("iou")
    m = cflearn.cv.CarefreePipeline(
        "cascade_u2net",
        {
            "in_channels": 3,
            "out_channels": 1,
            "lv1_model_ckpt_path": "pretrained/model_lite_finetuned.pt",
            "lite": True,
        },
        loss_name="multi_stage_iou",
        lr=1.0e-4,
        optimizer_name="sgd",
        scheduler_name="none",
        optimizer_config={
            "momentum": 0.9,
            "weight_decay": 5.0e-4,
            "nesterov": True,
        },
        loss_metrics_weights={"iou0": 1.0},
    )
    m.fit(train_loader, valid_loader, cuda="6")
