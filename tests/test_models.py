# type: ignore

import torch
import cflearn
import unittest

import numpy as np

from typing import Any
from torch.nn import Module
from cftool.array import allclose
from cflearn.constants import INPUT_KEY
from cflearn.constants import PREDICTIONS_KEY
from cflearn.misc.toolkit import to_numpy
from cflearn.misc.toolkit import to_torch
from cflearn.misc.toolkit import eval_context
from cflearn.models.ml.protocol import MERGED_KEY
from cflearn.models.ml.protocol import ONE_HOT_KEY
from cflearn.models.ml.protocol import EMBEDDING_KEY
from cflearn.models.ml.protocol import NUMERICAL_KEY


batch_size = 21
patch_size = 5
img_size = 35
in_channels = 13
out_channels = 5
num_downsample = 2
num_classes = 11
latent_dim = 111

inp = torch.randn(batch_size, in_channels, img_size, img_size)


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

    def test_nbm(self) -> None:
        shape = [3, 10]
        x = np.random.random(shape)
        y = np.random.random(shape)
        m = cflearn.ml.make_toy_model(model="nbm", data_tuple=(x, y))
        nbm = m.model.core.core
        net = to_torch(x)
        x_dims = (3,)
        y_dim = 7
        net1 = net
        res1 = to_numpy(nbm.inspect(net1, x_dims, y_dim))
        net2 = net[..., x_dims]
        res2 = to_numpy(nbm.inspect(net2, x_dims, y_dim, already_extracted=True))
        net3 = to_torch(np.random.random(shape))
        net3[..., x_dims] = net2
        net3 = net
        with eval_context(nbm):
            net3 = nbm(net3, return_features=True)
        net3 = net3[..., x_dims] * nbm.head.weight[..., x_dims][[y_dim]]
        res3 = to_numpy(net3)
        self.assertTrue(allclose(res1, res2, res3))

    def test_cv_clf_cnn(self) -> None:
        vanilla_clf = cflearn.VanillaClassifier(
            in_channels,
            num_classes,
            img_size,
            latent_dim,
            encoder1d_config={"num_downsample": num_downsample},
        )
        self.assertSequenceEqual(
            vanilla_clf.classify(inp)[PREDICTIONS_KEY].shape,
            [batch_size, num_classes],
        )

        for model_type in ["lite", "large"]:
            vgg_clf = cflearn.VanillaClassifier(
                in_channels,
                num_classes,
                img_size,
                512,
                encoder1d="backbone",
                encoder1d_config={"name": f"vgg19_{model_type}"},
            )
            self.assertSequenceEqual(
                vgg_clf.classify(inp)[PREDICTIONS_KEY].shape,
                [batch_size, num_classes],
            )

        resnet_clf = cflearn.VanillaClassifier(
            in_channels,
            num_classes,
            img_size,
            512,
            encoder1d="backbone",
            encoder1d_config={"name": "resnet18"},
        )
        self.assertSequenceEqual(
            resnet_clf.classify(inp)[PREDICTIONS_KEY].shape,
            [batch_size, num_classes],
        )

        resnet_zoo_clf = cflearn.DLZoo.load_model(
            "clf/resnet101",
            model_config=dict(
                in_channels=in_channels,
                num_classes=num_classes,
                img_size=img_size,
            ),
        )
        self.assertSequenceEqual(
            resnet_zoo_clf.classify(inp)[PREDICTIONS_KEY].shape,
            [batch_size, num_classes],
        )

        mobilenet_clf = cflearn.VanillaClassifier(
            in_channels,
            num_classes,
            img_size,
            320,
            encoder1d="backbone",
            encoder1d_config={"name": "mobilenet_v2"},
        )
        self.assertSequenceEqual(
            mobilenet_clf.classify(inp)[PREDICTIONS_KEY].shape,
            [batch_size, num_classes],
        )

    def test_cv_clf_mixed_stack(self) -> None:
        vit_clf = cflearn.VanillaClassifier(
            in_channels,
            num_classes,
            img_size,
            latent_dim,
            encoder1d="vit",
            encoder1d_config={
                "patch_size": patch_size,
                "attention_kwargs": {"embed_dim": 18},
            },
        )
        self.assertSequenceEqual(
            vit_clf.classify(inp)[PREDICTIONS_KEY].shape,
            [batch_size, num_classes],
        )

        cct_zoo_clf = cflearn.DLZoo.load_model(
            "clf/cct.large",
            model_config=dict(
                in_channels=in_channels,
                num_classes=num_classes,
                img_size=img_size,
            ),
        )
        self.assertSequenceEqual(
            cct_zoo_clf.classify(inp)[PREDICTIONS_KEY].shape,
            [batch_size, num_classes],
        )

        mixer_clf = cflearn.VanillaClassifier(
            in_channels,
            num_classes,
            img_size,
            latent_dim,
            encoder1d="mixer",
            encoder1d_config={"patch_size": patch_size},
        )
        self.assertSequenceEqual(
            mixer_clf.classify(inp)[PREDICTIONS_KEY].shape,
            [batch_size, num_classes],
        )

        fnet_clf = cflearn.VanillaClassifier(
            in_channels,
            num_classes,
            img_size,
            latent_dim,
            encoder1d="fnet",
            encoder1d_config={"patch_size": patch_size},
        )
        self.assertSequenceEqual(
            fnet_clf.classify(inp)[PREDICTIONS_KEY].shape,
            [batch_size, num_classes],
        )

    def test_cv_vae(self) -> None:
        vae1d = cflearn.VanillaVAE1D(in_channels, out_channels, img_size=img_size)
        self.assertSequenceEqual(
            vae1d.decode(torch.randn(batch_size, vae1d.latent_dim), labels=None).shape,
            [batch_size, out_channels, img_size, img_size],
        )

        vae2d = cflearn.VanillaVAE2D(in_channels, out_channels, img_size=img_size)
        assert vae2d.latent_resolution is not None
        latent_channels = vae2d.generator.decoder.latent_channels
        latent_d = latent_channels * vae2d.latent_resolution**2
        self.assertSequenceEqual(
            vae2d.decode(torch.randn(batch_size, latent_d), labels=None).shape,
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
            vq_vae.decode(
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

    def test_cv_gan(self) -> None:
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

    def test_cv_segmentation(self) -> None:
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
            u2net(
                torch.randn(
                    batch_size,
                    in_channels,
                    img_size,
                    img_size,
                )
            )[0].shape,
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
                ),
                torch.randn(
                    batch_size,
                    1,
                    img_size,
                    img_size,
                ),
            ).shape,
            [batch_size, out_channels, img_size, img_size],
        )

        cascade_u2net = cflearn.CascadeU2Net(
            in_channels,
            1,
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
            [batch_size, 1, img_size, img_size],
        )

        seg_former = cflearn.LinearSegmentation(in_channels, out_channels)
        self.assertSequenceEqual(
            seg_former.generate_from(
                torch.randn(
                    batch_size,
                    in_channels,
                    img_size,
                    img_size,
                )
            ).shape,
            [batch_size, out_channels, img_size, img_size],
        )

    def test_cv_style_transfer(self) -> None:
        adain_stylizer = cflearn.AdaINStylizer(in_channels)
        self.assertSequenceEqual(
            adain_stylizer.stylize(
                torch.randn(
                    batch_size,
                    in_channels,
                    img_size,
                    img_size,
                ),
                torch.randn(
                    batch_size,
                    in_channels,
                    img_size,
                    img_size,
                ),
            ).shape,
            [batch_size, in_channels, img_size, img_size],
        )

    def test_cv_generator(self) -> None:
        size = 3
        pixel_cnn = cflearn.PixelCNN(1, num_classes)
        self.assertSequenceEqual(pixel_cnn.sample(5, size).shape, [5, 1, size, size])

        vqgan_generator = cflearn.VQGANGenerator(
            img_size,
            123,
            in_channels,
            out_channels,
        )
        self.assertSequenceEqual(
            vqgan_generator(0, {INPUT_KEY: inp})[PREDICTIONS_KEY].shape,
            [batch_size, out_channels, img_size, img_size],
        )

        style_gan2_generator = cflearn.StyleGAN2Generator(32, 123)
        self.assertSequenceEqual(
            style_gan2_generator.generate_from(torch.randn(5, 123)).shape,
            [5, 3, 32, 32],
        )

        cycle_gan_generator = cflearn.CycleGANGenerator(in_channels, out_channels)
        self.assertSequenceEqual(
            cycle_gan_generator.generate_from(inp).shape,
            [batch_size, out_channels, img_size, img_size],
        )

    def test_cv_perceiver_io(self) -> None:
        perceiver_clf = cflearn.VanillaClassifier(
            in_channels,
            num_classes,
            img_size,
            16,
            encoder1d="perceiver_io",
            encoder1d_config={"patch_size": patch_size},
        )
        self.assertSequenceEqual(
            perceiver_clf.classify(inp)[PREDICTIONS_KEY].shape,
            [batch_size, num_classes],
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
