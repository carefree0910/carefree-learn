# type: ignore

import cflearn


train_loader, valid_loader = cflearn.cv.get_mnist(transform="for_generation")

m = cflearn.cv.CarefreePipeline(
    "gan",
    {"img_size": 28, "in_channels": 1},
    callback_names="generator",
    optimizer_settings={
        "g_parameters": {
            "optimizer": "adam",
            "scheduler": "warmup",
        },
        "d_parameters": {
            "optimizer": "adam",
            "scheduler": "warmup",
        },
    },
)
m.fit(train_loader, valid_loader, cuda="1")
