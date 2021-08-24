import torch
import cflearn
import unittest

from typing import Any
from torch.nn import Module
from cflearn.constants import INPUT_KEY
from cflearn.constants import PREDICTIONS_KEY
from cflearn.models.ml.protocol import MERGED_KEY
from cflearn.models.ml.protocol import ONE_HOT_KEY
from cflearn.models.ml.protocol import EMBEDDING_KEY
from cflearn.models.ml.protocol import NUMERICAL_KEY


class TestModels(unittest.TestCase):
    def test_ml(self) -> None:
        def _make(core_name: str, **kwargs: Any) -> Module:
            class _(Module):
                def __init__(self) -> None:
                    super().__init__()
                    _num_history = kwargs.pop("num_history", 1)
                    self.net = cflearn.MLModel(
                        in_dim,
                        out_dim,
                        _num_history,
                        encoder=None,
                        numerical_columns_mapping={i: i for i in range(in_dim)},
                        categorical_columns_mapping={},
                        use_one_hot=False,
                        use_embedding=False,
                        only_categorical=False,
                        core_name=core_name,
                        core_config=kwargs,
                        pre_process_batch=False,
                    )
                    self.net._init_with_trainer(None)

                def forward(self, net: torch.Tensor) -> torch.Tensor:
                    batch = {INPUT_KEY: net, MERGED_KEY: net, NUMERICAL_KEY: net}
                    for key in [ONE_HOT_KEY, EMBEDDING_KEY]:
                        batch[key] = None
                    return self.net(0, batch)[PREDICTIONS_KEY]

            if core_name == "ddr":
                kwargs["y_min_max"] = -1.0, 1.0
            return _()

        def _ts(core_name: str, **kwargs: Any) -> Module:
            kwargs["num_history"] = num_history
            return _make(core_name, **kwargs)

        in_dim = 7
        out_dim = 5
        batch_size = 21
        num_history = 11

        inp = torch.randn(batch_size, in_dim)
        ts = torch.randn(batch_size, num_history, in_dim)

        self.assertSequenceEqual(_make("linear")(inp).shape, [batch_size, out_dim])
        self.assertSequenceEqual(_ts("linear")(ts).shape, [batch_size, out_dim])
        self.assertSequenceEqual(_make("fcnn")(inp).shape, [batch_size, out_dim])
        self.assertSequenceEqual(_ts("fcnn")(ts).shape, [batch_size, out_dim])
        self.assertSequenceEqual(_make("wnd")(inp).shape, [batch_size, out_dim])
        self.assertSequenceEqual(_ts("wnd")(ts).shape, [batch_size, out_dim])
        self.assertSequenceEqual(_make("ddr")(inp).shape, [batch_size, out_dim])
        self.assertSequenceEqual(_ts("ddr")(ts).shape, [batch_size, out_dim])
        self.assertSequenceEqual(_make("rnn")(ts).shape, [batch_size, out_dim])
        self.assertSequenceEqual(_ts("rnn")(ts).shape, [batch_size, out_dim])
        self.assertSequenceEqual(_ts("transformer")(ts).shape, [batch_size, out_dim])
        self.assertSequenceEqual(_ts("mixer")(ts).shape, [batch_size, out_dim])
        self.assertSequenceEqual(_ts("fnet")(ts).shape, [batch_size, out_dim])

    def test_cv(self) -> None:
        batch_size = 21
        patch_size = 5
        img_size = 15
        in_channels = 13
        out_channels = 5
        num_downsample = 2
        num_classes = 11
        latent_dim = 111

        inp = torch.randn(batch_size, in_channels, img_size, img_size)

        vanilla_clf = cflearn.VanillaClassifier(
            in_channels,
            num_classes,
            img_size,
            latent_dim,
            encoder1d_configs={"num_downsample": num_downsample},
        )
        self.assertSequenceEqual(
            vanilla_clf.classify(inp).shape,
            [batch_size, num_classes],
        )

        resnet_clf = cflearn.VanillaClassifier(
            in_channels,
            num_classes,
            img_size,
            512,
            encoder1d="backbone",
            encoder1d_configs={"name": "resnet18"},
        )
        self.assertSequenceEqual(
            resnet_clf.classify(inp).shape,
            [batch_size, num_classes],
        )

        mobilenet_clf = cflearn.VanillaClassifier(
            in_channels,
            num_classes,
            img_size,
            320,
            encoder1d="backbone",
            encoder1d_configs={"name": "mobilenet_v2"},
        )
        self.assertSequenceEqual(
            mobilenet_clf.classify(inp).shape,
            [batch_size, num_classes],
        )

        vit_clf = cflearn.VanillaClassifier(
            in_channels,
            num_classes,
            img_size,
            latent_dim,
            encoder1d="vit",
            encoder1d_configs={"patch_size": patch_size, "embed_dim": 18},
        )
        self.assertSequenceEqual(vit_clf.classify(inp).shape, [batch_size, num_classes])

        mixer_clf = cflearn.VanillaClassifier(
            in_channels,
            num_classes,
            img_size,
            latent_dim,
            encoder1d="mixer",
            encoder1d_configs={"patch_size": patch_size},
        )
        self.assertSequenceEqual(
            mixer_clf.classify(inp).shape,
            [batch_size, num_classes],
        )

        fnet_clf = cflearn.VanillaClassifier(
            in_channels,
            num_classes,
            img_size,
            latent_dim,
            encoder1d="fnet",
            encoder1d_configs={"patch_size": patch_size},
        )
        self.assertSequenceEqual(
            fnet_clf.classify(inp).shape,
            [batch_size, num_classes],
        )

        perceiver_clf = cflearn.VanillaClassifier(
            in_channels,
            num_classes,
            img_size,
            16,
            encoder1d="perceiver_io",
            encoder1d_configs={"patch_size": patch_size},
        )
        self.assertSequenceEqual(
            perceiver_clf.classify(inp).shape,
            [batch_size, num_classes],
        )

        vae = cflearn.VanillaVAE(in_channels, out_channels, img_size=img_size)
        self.assertSequenceEqual(
            vae.decode(torch.randn(batch_size, vae.latent_dim), labels=None).shape,
            [batch_size, out_channels, img_size, img_size],
        )

        siren_vae = cflearn.SirenVAE(img_size, in_channels, out_channels)
        self.assertSequenceEqual(
            siren_vae.decode(
                torch.randn(batch_size, siren_vae.latent_dim),
                labels=None,
            ).shape,
            [batch_size, out_channels, img_size, img_size],
        )

        num_code = 16
        vq_vae = cflearn.VQVAE(img_size, num_code, in_channels, out_channels)
        self.assertSequenceEqual(
            vq_vae._decode(
                torch.randn(
                    batch_size,
                    vq_vae.latent_channels,
                    vq_vae.latent_resolution,
                    vq_vae.latent_resolution,
                ),
                labels=None,
            ).shape,
            [batch_size, out_channels, img_size, img_size],
        )

        gan = cflearn.VanillaGAN(img_size, in_channels, out_channels)
        self.assertSequenceEqual(
            gan.decode(torch.randn(batch_size, gan.latent_dim), labels=None).shape,
            [batch_size, out_channels, img_size, img_size],
        )

        siren_gan = cflearn.SirenGAN(img_size, in_channels, out_channels)
        self.assertSequenceEqual(
            siren_gan.decode(
                torch.randn(batch_size, siren_gan.latent_dim),
                labels=None,
            ).shape,
            [batch_size, out_channels, img_size, img_size],
        )

        unet = cflearn.UNet(in_channels, out_channels)
        self.assertSequenceEqual(
            unet.generate_from(
                torch.randn(
                    batch_size,
                    in_channels,
                    img_size,
                    img_size,
                )
            ).shape,
            [batch_size, out_channels, img_size, img_size],
        )

        u2net = cflearn.U2Net(
            in_channels,
            out_channels,
            num_layers=2,
            num_inner_layers=3,
        )
        self.assertSequenceEqual(
            u2net.generate_from(
                torch.randn(
                    batch_size,
                    in_channels,
                    img_size,
                    img_size,
                )
            ).shape,
            [batch_size, out_channels, img_size, img_size],
        )

        alpha_refine = cflearn.AlphaRefineNet(in_channels, out_channels)
        self.assertSequenceEqual(
            alpha_refine.refine_from(
                torch.randn(
                    batch_size,
                    in_channels,
                    img_size,
                    img_size,
                )
            ).shape,
            [batch_size, out_channels, img_size, img_size],
        )

        cascade_u2net = cflearn.CascadeU2Net(
            in_channels,
            out_channels,
            num_layers=2,
            num_inner_layers=3,
        )
        self.assertSequenceEqual(
            cascade_u2net.generate_from(
                torch.randn(
                    batch_size,
                    in_channels,
                    img_size,
                    img_size,
                )
            ).shape,
            [batch_size, out_channels, img_size, img_size],
        )

        pixel_cnn = cflearn.PixelCNN(1, num_classes)
        self.assertSequenceEqual(
            pixel_cnn.sample(1, img_size).shape,
            [1, 1, img_size, img_size],
        )

        perceiver_io = cflearn.PerceiverIOGenerator(2, in_channels, out_channels)
        self.assertSequenceEqual(
            perceiver_io.generate_from(
                torch.randn(
                    batch_size,
                    in_channels,
                    img_size,
                    img_size,
                )
            ).shape,
            [batch_size, out_channels, img_size, img_size],
        )


if __name__ == "__main__":
    unittest.main()