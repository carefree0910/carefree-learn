import time
import torch
import unittest

import numpy as np
import torch.nn as nn

from typing import List
from typing import Tuple
from cflearn.schema import MLEncoderSettings
from cflearn.constants import LATENT_KEY
from cflearn.toolkit import eval_context
from cflearn.toolkit import inject_parameters
from cflearn.modules import get_latent_resolution
from cflearn.modules import BN
from cflearn.modules import EMA
from cflearn.modules import DNDF
from cflearn.modules import Conv2d
from cflearn.modules import Linear
from cflearn.modules import Lambda
from cflearn.modules import Attention
from cflearn.modules import BackboneEncoder
from cflearn.modules.core.ml_encoder import Encoder


class TestBlocks(unittest.TestCase):
    def test_lambda(self) -> None:
        inp = torch.randn(2, 3, 4, 5)
        self.assertTrue(
            torch.allclose(
                Lambda(lambda net: net + 1)(inp),
                inp + 1,
            )
        )

    def test_bn(self) -> None:
        bn = BN(128)
        self.assertSequenceEqual(bn(torch.randn(4, 128)).shape, [4, 128])
        self.assertSequenceEqual(bn(torch.randn(4, 8, 128)).shape, [4, 8, 128])

    def test_ema(self) -> None:
        decay = 0.9
        p1 = nn.Parameter(torch.randn(2, 3, 4, 5))
        p2 = nn.Parameter(torch.randn(2, 3, 4, 5))
        p3 = nn.Parameter(torch.randn(2, 3, 4, 5))
        gt = p1.data
        gt = decay * gt + (1.0 - decay) * p2.data
        gt = decay * gt + (1.0 - decay) * p3.data
        ema = EMA(decay, [("test", p1)])
        p1.data = p2.data
        ema()
        p1.data = p3.data
        ema()
        ema.eval()
        ema.train()
        ema.eval()
        ema.train()
        ema.eval()
        self.assertTrue(torch.allclose(p1.data, gt.data))
        ema.train()
        ema.eval()
        ema.train()
        ema.eval()
        ema.train()
        self.assertTrue(torch.allclose(p1.data, p3.data))
        ema.eval()
        ema.eval()
        ema.train()
        ema.train()
        ema.eval()
        ema.eval()
        self.assertTrue(torch.allclose(p1.data, gt.data))
        ema.train()
        ema.train()
        ema.eval()
        ema.eval()
        ema.train()
        ema.train()
        self.assertTrue(torch.allclose(p1.data, p3.data))
        with eval_context(ema):
            self.assertTrue(torch.allclose(p1.data, gt.data))
        self.assertTrue(torch.allclose(p1.data, p3.data))

    def test_dndf(self) -> None:
        input_dim = 256
        output_dim = 512
        batch_size = 32

        net = torch.randn(batch_size, input_dim)

        dndf = DNDF(input_dim, output_dim)
        probabilities = dndf(net)
        self.assertTrue(torch.allclose(probabilities.sum(1), torch.ones(batch_size)))

        dndf = DNDF(input_dim, None)
        features = dndf(net)
        self.assertTrue(torch.allclose(features.sum(2), torch.ones(features.shape[:2])))

    def test_fast_dndf(self) -> None:
        def loss_function(outputs: torch.Tensor) -> torch.Tensor:
            return -outputs[range(batch_size), labels].mean()

        d = 128
        batch_size = 1024

        for k in [1, 10, 20]:
            inp = torch.randn(batch_size, d, requires_grad=True)
            labels = torch.randint(k, [batch_size])

            def _run() -> Tuple[float, float]:
                dndf = DNDF(d, k, use_fast_dndf=False)
                net = torch.empty_like(inp).requires_grad_(True)
                net.data = inp.data

                t1 = time.time()
                loss = loss_function(dndf(net))
                loss.backward()
                g1 = net.grad
                t2 = time.time()

                dndf_fast = DNDF(d, k, use_fast_dndf=True)
                inject_parameters(dndf, dndf_fast)
                net = torch.empty_like(inp).requires_grad_(True)
                net.data = inp.data

                t3 = time.time()
                loss = loss_function(dndf_fast(net))
                loss.backward()
                g2 = net.grad
                t4 = time.time()

                self.assertTrue(torch.allclose(g1, g2))
                return t2 - t1, t4 - t3

            slow_ts, fast_ts = [], []
            for _ in range(10):
                slow_t, fast_t = _run()
                slow_ts.append(slow_t)
                fast_ts.append(fast_t)
            slow_ts_array, fast_ts_array = map(np.array, [slow_ts, fast_ts])
            score_fn = lambda arr: arr.mean().item() + arr.std().item()
            slow_t = score_fn(slow_ts_array)
            fast_t = score_fn(fast_ts_array)
            print(f"slow : {slow_t} ; fast : {fast_t}")
            self.assertTrue(fast_t < slow_t)

    def test_attention(self) -> None:
        num_heads = 8
        input_dim = 256
        batch_size = 32
        q_len = 20
        k_len = 40
        k_dim = 512
        v_dim = 1024

        q = torch.randn(batch_size, q_len, input_dim)
        k = torch.randn(batch_size, k_len, k_dim)
        v = torch.randn(batch_size, k_len, v_dim)
        mask = torch.rand(batch_size, q_len, k_len) >= 0.5

        torch_attention = nn.MultiheadAttention(
            input_dim,
            num_heads,
            kdim=k_dim,
            vdim=v_dim,
        )
        permute = lambda t: t.permute(1, 0, 2)
        qt, kt, vt = map(permute, [q, k, v])
        torch_attn_mask = mask.repeat(num_heads, 1, 1)
        torch_output = torch_attention(qt, kt, vt, attn_mask=torch_attn_mask)[0]

        attention = Attention(input_dim, num_heads, k_dim=k_dim, v_dim=v_dim)
        inject_parameters(torch_attention, attention)
        output = attention(q, k, v, mask=mask).output
        torch_output = permute(torch_output)
        self.assertTrue(torch.allclose(torch_output, output, atol=1.0e-4))

    def test_linear(self) -> None:
        input_dim = 256
        output_dim = 512
        batch_size = 32

        net = torch.randn(batch_size, input_dim)
        torch_linear = nn.Linear(input_dim, output_dim)
        torch_output = torch_linear(net)

        linear = Linear(input_dim, output_dim)
        inject_parameters(torch_linear, linear)
        output = linear(net)

        self.assertTrue(torch.allclose(torch_output, output))

    def test_conv2d(self) -> None:
        batch_size = 32
        h = w = 32
        in_channels = 16
        out_channels = 128
        kernel_size = 4
        stride = 2
        padding = 1
        dilation = 2
        groups = 2

        net = torch.randn(batch_size, in_channels, h, w)
        torch_conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
        )
        torch_output = torch_conv2d(net)

        conv2d = Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            groups=groups,
            stride=stride,
            dilation=dilation,
            padding=padding,
        )
        inject_parameters(torch_conv2d, conv2d)
        output = conv2d(net)

        self.assertTrue(torch.allclose(torch_output, output))

    def test_ml_embedding(self) -> None:
        def _make(embed_dims: List[int]) -> Encoder:
            nd = len(embed_dims)
            return Encoder(
                {
                    str(idx): MLEncoderSettings(
                        dim=embed_dims[idx],
                        methods="embedding",
                        method_configs={"out_dim": embed_dims[idx]},
                    )
                    for idx in range(nd)
                },
            )

        def _run(
            encoder: Encoder,
            net: torch.Tensor,
            repeat: int,
        ) -> Tuple[torch.Tensor, float]:
            t = time.time()
            encoded = encoder(net).embedding
            for _ in range(repeat - 1):
                encoder(net)
            return encoded, time.time() - t

        def _test_case(embed_dims: List[int]) -> None:
            net = torch.cat([torch.randint(d, (bs, 1)) for d in embed_dims], dim=1)
            e = _make(embed_dims)
            r1, _ = _run(e, net, num_repeat)
            self.assertSequenceEqual(r1.shape, [bs, sum(embed_dims)])

        bs = 128
        num_repeat = 100
        _test_case([4] * 30)
        _test_case([4] * 10 + [8] * 10 + [16] * 10)
        for dim in [4, 64, 128, 256]:
            for n_dim in [8, 16]:
                _test_case([dim] * n_dim)
        for dim in [4, 16]:
            dims = [dim] * 16 + [dim * 2] * 12 + [dim * 3] * 8
            _test_case(dims)

    def test_cv_backbone(self) -> None:
        def _check(name: str) -> None:
            is_rep_vgg = name.startswith("rep_vgg")
            if not is_rep_vgg:
                key = name
                check_rep_vgg_deploy = False
            else:
                check_rep_vgg_deploy = name.endswith("_deploy")
                if not check_rep_vgg_deploy:
                    key = name
                else:
                    key = "_".join(name.split("_")[:-1])
            encoder = BackboneEncoder(key, in_channels)
            results = encoder(inp)
            backbone = encoder.net
            if check_rep_vgg_deploy:
                backbone.original.switch_to_deploy()
            return_nodes = list(backbone.return_nodes.values())
            latent_resolution = get_latent_resolution(encoder, img_size)
            for i, (k, v) in enumerate(results.items()):
                if k == LATENT_KEY:
                    self.assertEqual(v.shape[1], backbone.latent_channels)
                    self.assertEqual(v.shape[-1], latent_resolution)
                else:
                    self.assertEqual(k, return_nodes[i])
                    self.assertEqual(v.shape[1], backbone.out_channels[i])

        img_size = 37
        batch_size = 11
        in_channels = 3

        inp = torch.randn(batch_size, in_channels, img_size, img_size)
        list(
            map(
                _check,
                [
                    "mobilenet_v2",
                    "vgg16",
                    "vgg19",
                    "vgg19_lite",
                    "vgg19_large",
                    "vgg_style",
                    "rep_vgg",
                    "rep_vgg_deploy",
                    "rep_vgg_lite",
                    "rep_vgg_lite_deploy",
                    "rep_vgg_large",
                    "rep_vgg_large_deploy",
                    "mix_vit",
                    "mix_vit_lite",
                    "mix_vit_large",
                    "resnet18",
                    "resnet50",
                    "resnet101",
                    "resnet152",
                ],
            )
        )


if __name__ == "__main__":
    unittest.main()
