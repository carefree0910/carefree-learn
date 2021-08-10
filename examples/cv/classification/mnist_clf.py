# type: ignore

import cflearn


train_loader, valid_loader = cflearn.cv.get_mnist(transform="to_tensor")

m = cflearn.cv.CarefreePipeline(
    "clf",
    {
        # "img_size": 28,
        "in_channels": 1,
        "num_classes": 10,
        "latent_dim": 512,
        "encoder1d": "backbone",
        "encoder1d_configs": {"name": "resnet18"},
    },
    loss_name="cross_entropy",
    metric_names="acc",
)
m.fit(train_loader, valid_loader, cuda="0")
