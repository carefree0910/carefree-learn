import torch
import unittest

import torch.nn as nn

from cflearn.modules.blocks import *


class TestBlocks(unittest.TestCase):
    def test_linear(self) -> None:
        input_dim = 256
        output_dim = 512
        batch_size = 32

        net = torch.randn(batch_size, input_dim)
        weight = torch.randn(output_dim, input_dim)
        bias = torch.randn(output_dim)

        torch_linear = nn.Linear(input_dim, output_dim)
        torch_linear.weight.data = weight
        torch_linear.bias.data = bias
        torch_output = torch_linear(net)

        linear = Linear(input_dim, output_dim)
        linear.weight.data = weight
        assert linear.bias is not None
        linear.bias.data = bias
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

    def test_invertible(self) -> None:
        dim = 512
        h_dim = int(dim // 2)
        batch_size = 32

        net1 = torch.randn(batch_size, h_dim)
        net2 = torch.randn(batch_size, h_dim)
        invertible = InvertibleBlock(dim)
        o1, o2 = invertible(net1, net2)
        r1, r2 = invertible.inverse(o1, o2)
        self.assertTrue(torch.allclose(net1, r1, rtol=1e-4, atol=1e-4))
        self.assertTrue(torch.allclose(net2, r2, rtol=1e-4, atol=1e-4))

        res_invertible = ResInvertibleBlock(dim)
        o1, o2 = res_invertible(net1, net2)
        r1, r2 = res_invertible.inverse(o1, o2)
        self.assertTrue(torch.allclose(o1, net1 + net2, rtol=1e-4, atol=1e-4))
        self.assertTrue(torch.allclose(o2, net1 + 2.0 * net2, rtol=1e-4, atol=1e-4))
        self.assertTrue(torch.allclose(net1, r1, rtol=1e-4, atol=1e-4))
        self.assertTrue(torch.allclose(net2, r2, rtol=1e-4, atol=1e-4))

    def test_attention(self) -> None:
        num_heads = 8
        input_dim = embed_dim = 256
        batch_size = 32
        q_len = 20
        k_len = 40
        k_dim = 512
        v_dim = 1024

        q = torch.randn(batch_size, q_len, input_dim)
        k = torch.randn(batch_size, k_len, k_dim)
        v = torch.randn(batch_size, k_len, v_dim)
        mask = torch.randn(batch_size, q_len, k_len) >= 0.5

        q_proj_weight = torch.randn(embed_dim, input_dim)
        k_proj_weight = torch.randn(embed_dim, k_dim)
        v_proj_weight = torch.randn(embed_dim, v_dim)
        in_proj_bias = torch.randn(3 * embed_dim)
        out_proj_weight = torch.randn(embed_dim, input_dim)
        out_proj_bias = torch.randn(input_dim)

        torch_attention = nn.MultiheadAttention(
            input_dim,
            num_heads,
            kdim=k_dim,
            vdim=v_dim,
        )
        torch_attention.q_proj_weight.data = q_proj_weight
        torch_attention.k_proj_weight.data = k_proj_weight
        torch_attention.v_proj_weight.data = v_proj_weight
        torch_attention.in_proj_bias.data = in_proj_bias
        torch_attention.out_proj.weight.data = out_proj_weight
        torch_attention.out_proj.bias.data = out_proj_bias

        permute = lambda t: t.permute(1, 0, 2)
        qt, kt, vt = map(permute, [q, k, v])
        torch_attn_mask = mask.repeat(num_heads, 1, 1)
        torch_output = torch_attention(qt, kt, vt, attn_mask=torch_attn_mask)[0]

        attention = Attention(input_dim, num_heads)
        qb, kb, vb = in_proj_bias.split(input_dim)
        attention.q_linear.linear.weight.data = q_proj_weight
        attention.q_linear.linear.bias.data = qb
        attention.k_linear.linear.weight.data = k_proj_weight
        attention.k_linear.linear.bias.data = kb
        attention.v_linear.linear.weight.data = v_proj_weight
        attention.v_linear.linear.bias.data = vb
        attention.out_linear.linear.weight.data = out_proj_weight
        attention.out_linear.linear.bias.data = out_proj_bias

        output = attention(q, k, v, mask=mask).output

        self.assertTrue(torch.allclose(permute(torch_output), output))


if __name__ == "__main__":
    unittest.main()
