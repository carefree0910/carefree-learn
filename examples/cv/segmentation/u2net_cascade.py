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
            "lv1_model_ckpt_path": "pretrained/model_269926.pt",
            "lv2_model_config": {"lite": True},
            "lite": False,
        },
        loss_name="multi_stage_iou",
        scheduler_name="none",
        loss_metrics_weights={"iou0": 1.0},
    )
    m.fit(train_loader, valid_loader, cuda="6")
