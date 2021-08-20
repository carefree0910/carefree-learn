import time
import torch
import unittest

import torch.nn as nn

from cflearn.misc.toolkit import inject_parameters
from cflearn.modules.blocks import DNDF
from cflearn.modules.blocks import Linear
from cflearn.modules.blocks import Attention


class TestBlocks(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
