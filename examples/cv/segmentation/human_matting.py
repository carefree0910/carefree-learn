import os
import cv2
import cflearn

import numpy as np

from typing import List
from cflearn.misc.toolkit import to_device
from cflearn.misc.toolkit import save_images
from cflearn.misc.toolkit import eval_context


src_folder = "data"
tgt_folder = "human_matting"
label_folder = "human_matting_labels"


def prepare() -> None:
    def label_fn(hierarchy: List[str]) -> str:
        matting_path = os.path.abspath(
            os.path.join(
                hierarchy[0],
                "matting",
                hierarchy[2],
                hierarchy[3].replace("clip", "matting"),
                hierarchy[4].replace(".jpg", ".png"),
            )
        )
        alpha = cv2.imread(matting_path, cv2.IMREAD_UNCHANGED)[..., -1:]
        alpha = (alpha > 10).astype(np.int64).transpose([2, 0, 1])
        file = hierarchy[-1]
        file_id = os.path.splitext(file)[0]
        os.makedirs(label_folder, exist_ok=True)
        label_path = os.path.join(label_folder, f"{file_id}.npy")
        label_path = os.path.abspath(label_path)
        if os.path.isfile(label_path):
            return label_path
        np.save(label_path, alpha)
        return label_path

    def filter_fn(hierarchy: List[str]) -> bool:
        return hierarchy[1] == "clip_img"

    cflearn.cv.prepare_image_folder(
        src_folder,
        tgt_folder,
        to_index=False,
        label_fn=label_fn,
        filter_fn=filter_fn,
        make_labels_in_parallel=True,
    )


@cflearn.ArtifactCallback.register("unet")
class UnetCallback(cflearn.ArtifactCallback):
    key = "images"

    def log_artifacts(self, trainer: cflearn.Trainer) -> None:
        if not self.is_rank_0:
            return None
        batch = next(iter(trainer.validation_loader))
        batch = to_device(batch, trainer.device)
        original = batch[cflearn.INPUT_KEY]
        with eval_context(trainer.model):
            logits = trainer.model.generate_from(original)
            seg_map = logits.argmax(1, keepdim=True).float()
        image_folder = self._prepare_folder(trainer)
        save_images(original, os.path.join(image_folder, "original.png"))
        label = batch[cflearn.LABEL_KEY].float()
        save_images(label, os.path.join(image_folder, "label.png"))
        save_images(seg_map, os.path.join(image_folder, "segmentation.png"))


if __name__ == "__main__":
    prepare()
    train_loader, valid_loader = cflearn.cv.get_image_folder_loaders(
        tgt_folder,
        batch_size=4,
        num_workers=0,
        transform="to_tensor",
    )
    m = cflearn.cv.CarefreePipeline(
        "unet",
        {
            "in_channels": 3,
            "out_channels": 2,
        },
        loss_name="cross_entropy",
        metric_names="acc",
        tqdm_settings={"use_tqdm": True},
    )
    m.fit(train_loader, valid_loader, cuda="1")
