import time
import torch
import unittest

import torch.nn as nn

from typing import List
from typing import Tuple
from cflearn.misc.toolkit import inject_parameters
from cflearn.modules.blocks import BN
from cflearn.modules.blocks import EMA
from cflearn.modules.blocks import DNDF
from cflearn.modules.blocks import Conv2d
from cflearn.modules.blocks import Linear
from cflearn.modules.blocks import Lambda
from cflearn.modules.blocks import Attention
from cflearn.models.ml.encoders import Encoder


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
        self.assertTrue(torch.allclose(p1.data, gt.data))
        ema.train()
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

        for k in [1, 1, 10]:
            inp = torch.randn(batch_size, d, requires_grad=True)
            labels = torch.randint(k, [batch_size])

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
            slow_t, fast_t = t2 - t1, t4 - t3
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
        mask = torch.randn(batch_size, q_len, k_len) >= 0.5

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
        def _make(
            embed_dims: List[int],
            use_fast_embedding: bool,
            recover_original_dim: bool,
        ) -> Encoder:
            nd = len(embed_dims)
            return Encoder(
                {
                    "use_fast_embedding": use_fast_embedding,
                    "recover_original_dim": recover_original_dim,
                },
                embed_dims,
                ["embedding"] * nd,
                [{"embedding_dim": d} for d in embed_dims],
                list(range(nd)),
                [],
            )

        def _run(
            encoder: Encoder,
            net: torch.Tensor,
            repeat: int,
        ) -> Tuple[torch.Tensor, float]:
            t = time.time()
            encoded = encoder(net, None, None).embedding
            for _ in range(repeat - 1):
                encoder(net, None, None)
            return encoded, time.time() - t

        def _test_case(embed_dims: List[int]) -> Tuple[float, float, float]:
            nd = len(embed_dims)
            net = torch.cat([torch.randint(d, (bs, 1)) for d in embed_dims], dim=1)
            e = _make(embed_dims, False, False)
            r1, t1_ = _run(e, net, num_repeat)
            self.assertSequenceEqual(r1.shape, [bs, sum(embed_dims)])
            e = _make(embed_dims, True, False)
            r2, t2_ = _run(e, net, num_repeat)
            self.assertSequenceEqual(r2.shape, [bs, nd * max(embed_dims)])
            e = _make(embed_dims, True, True)
            r3, t3_ = _run(e, net, num_repeat)
            self.assertSequenceEqual(r3.shape, [bs, sum(embed_dims)])
            return t1_, t2_, t3_

        bs = 128
        num_repeat = 100
        t1, t2, t3 = _test_case([4] * 30)
        self.assertTrue(t1 > t2 and t1 > t3)
        t1, t2, t3 = _test_case([4] * 10 + [8] * 10 + [16] * 10)
        self.assertTrue(t1 > t3 > t2)
        for dim in [4, 64, 128, 256]:
            for n_dim in [10, 3]:
                t1, t2, t3 = _test_case([dim] * n_dim)
                self.assertTrue(t1 > t2 and t1 > t3)
        for dim in [4, 16]:
            dims = [dim] * 8 + [dim * 2] * 8 + [dim * 3] * 8
            t1, t2, t3 = _test_case(dims)
            self.assertTrue(t1 > t3 > t2)


if __name__ == "__main__":
    unittest.main()
