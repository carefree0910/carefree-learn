# type: ignore

import cflearn


num_classes = 10
train_loader, valid_loader = cflearn.cv.get_mnist(transform="for_generation")

m = cflearn.cv.CarefreePipeline(
    "vae",
    {
        "img_size": 28,
        "in_channels": 1,
        "num_classes": num_classes,
    },
    callback_names="generator",
    loss_metrics_weights={"kld": 0.001, "mse": 1.0},
)
m.fit(train_loader, valid_loader, cuda="1")
