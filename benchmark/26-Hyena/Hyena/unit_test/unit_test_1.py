import unittest
import random

# ------------------------------------------------------------------------------
# Either paste the entire Hyena code here or import it if it's in another module:
#
# from your_hyena_module import HyenaOperator
#
# For clarity in this snippet, we'll assume you have the HyenaOperator code
# in a file "hyena.py" and do:
#
# from hyena import HyenaOperator
#
# ------------------------------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


import logging
from pathlib import Path

# Set up logging to capture test pass/fail results
log_dir = Path(__file__).resolve().parent / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / 'unit_test_1.log'
logging.basicConfig(filename=log_file,
                    filemode="w",
                    level=logging.INFO,
                    format="%(asctime)s - %(message)s")


def fftconv(u, k, D):
    seqlen = u.shape[-1]
    fft_size = 2 * seqlen

    k_f = torch.fft.rfft(k, n=fft_size) / fft_size
    u_f = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size)

    if len(u.shape) > 3: k_f = k_f.unsqueeze(1)
    y = torch.fft.irfft(u_f * k_f, n=fft_size, norm='forward')[..., :seqlen]

    out = y + u * D.unsqueeze(-1)
    return out.to(dtype=u.dtype)


@torch.jit.script
def mul_sum(q, y):
    return (q * y).sum(dim=1)


class OptimModule(nn.Module):
    """ Interface for Module that allows registering buffers/parameters with configurable optimizer hyperparameters """

    def register(self, name, tensor, lr=None, wd=0.0):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {}
            if lr is not None: optim["lr"] = lr
            if wd is not None: optim["weight_decay"] = wd
            setattr(getattr(self, name), "_optim", optim)


class Sin(nn.Module):
    def __init__(self, dim, w=10, train_freq=True):
        super().__init__()
        self.freq = nn.Parameter(w * torch.ones(1, dim)) if train_freq else w * torch.ones(1, dim)

    def forward(self, x):
        return torch.sin(self.freq * x)


class PositionalEmbedding(OptimModule):
    def __init__(self, emb_dim: int, seq_len: int, lr_pos_emb: float = 1e-5, **kwargs):
        """Complex exponential positional embeddings for Hyena filters."""
        super().__init__()

        self.seq_len = seq_len
        # The time embedding fed to the filteres is normalized so that t_f = 1
        t = torch.linspace(0, 1, self.seq_len)[None, :, None]  # 1, L, 1

        if emb_dim > 1:
            bands = (emb_dim - 1) // 2
            # To compute the right embeddings we use the "proper" linspace
        t_rescaled = torch.linspace(0, seq_len - 1, seq_len)[None, :, None]
        w = 2 * math.pi * t_rescaled / seq_len  # 1, L, 1

        f = torch.linspace(1e-4, bands - 1, bands)[None, None]
        z = torch.exp(-1j * f * w)
        z = torch.cat([t, z.real, z.imag], dim=-1)
        self.register("z", z, lr=lr_pos_emb)
        self.register("t", t, lr=0.0)

    def forward(self, L):
        return self.z[:, :L], self.t[:, :L]


class ExponentialModulation(OptimModule):
    def __init__(
            self,
            d_model,
            fast_decay_pct=0.3,
            slow_decay_pct=1.5,
            target=1e-2,
            modulation_lr=0.0,
            modulate: bool = True,
            shift: float = 0.0,
            **kwargs
    ):
        super().__init__()
        self.modulate = modulate
        self.shift = shift
        max_decay = math.log(target) / fast_decay_pct
        min_decay = math.log(target) / slow_decay_pct
        deltas = torch.linspace(min_decay, max_decay, d_model)[None, None]
        self.register("deltas", deltas, lr=modulation_lr)

    def forward(self, t, x):
        if self.modulate:
            decay = torch.exp(-t * self.deltas.abs())
            x = x * (decay + self.shift)
        return x


class HyenaFilter(OptimModule):
    def __init__(
            self,
            d_model,
            emb_dim=3,  # dim of input to MLP, augments with positional encoding
            order=16,  # width of the implicit MLP
            fused_fft_conv=False,
            seq_len=1024,
            lr=1e-3,
            lr_pos_emb=1e-5,
            dropout=0.0,
            w=1,  # frequency of periodic activations
            wd=0,  # weight decay of kernel parameters
            bias=True,
            num_inner_mlps=2,
            normalized=False,
            **kwargs
    ):
        """
        Implicit long filter with modulation.

        Args:
            d_model: number of channels in the input
            emb_dim: dimension of the positional encoding (`emb_dim` - 1) // 2 is the number of bands
            order: width of the FFN
            num_inner_mlps: number of inner linear layers inside filter MLP
        """
        super().__init__()
        self.d_model = d_model
        self.use_bias = bias
        self.fused_fft_conv = fused_fft_conv
        self.bias = nn.Parameter(torch.randn(self.d_model))
        self.dropout = nn.Dropout(dropout)

        act = Sin(dim=order, w=w)
        self.emb_dim = emb_dim
        assert emb_dim % 2 != 0 and emb_dim >= 3, "emb_dim must be odd and greater or equal to 3 (time, sine and cosine)"
        self.seq_len = seq_len

        self.pos_emb = PositionalEmbedding(emb_dim, seq_len, lr_pos_emb)

        self.implicit_filter = nn.Sequential(
            nn.Linear(emb_dim, order),
            act,
        )
        for i in range(num_inner_mlps):
            self.implicit_filter.append(nn.Linear(order, order))
            self.implicit_filter.append(act)

        self.implicit_filter.append(nn.Linear(order, d_model, bias=False))

        self.modulation = ExponentialModulation(d_model, **kwargs)

        self.normalized = normalized
        for c in self.implicit_filter.children():
            for name, v in c.state_dict().items():
                optim = {"weight_decay": wd, "lr": lr}
                setattr(getattr(c, name), "_optim", optim)

    def filter(self, L, *args, **kwargs):
        z, t = self.pos_emb(L)
        h = self.implicit_filter(z)
        h = self.modulation(t, h)
        return h

    def forward(self, x, L, k=None, bias=None, *args, **kwargs):
        if k is None: k = self.filter(L)

        # Ensure compatibility with filters that return a tuple
        k = k[0] if type(k) is tuple else k

        y = fftconv(x, k, bias)
        return y


class HyenaOperator(nn.Module):
    def __init__(
            self,
            d_model,
            l_max,
            order=2,
            filter_order=64,
            dropout=0.0,
            filter_dropout=0.0,
            **filter_args,
    ):
        r"""
        Hyena operator described in the paper https://arxiv.org/pdf/2302.10866.pdf

        Args:
            d_model (int): Dimension of the input and output embeddings (width of the layer)
            l_max: (int): Maximum input sequence length. Defaults to None
            order: (int): Depth of the Hyena recurrence. Defaults to 2
            dropout: (float): Dropout probability. Defaults to 0.0
            filter_dropout: (float): Dropout probability for the filter. Defaults to 0.0
        """
        super().__init__()
        # TODO: Implement code here
        pass

    def forward(self, u, *args, **kwargs):
        # TODO: Implement code here
        pass


class TestHyenaOperator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # We fix the random seeds for reproducibility (so we generate the exact same
        # test input in both your reference code and the tested code).
        random.seed(0)
        torch.manual_seed(0)

        # We'll prepare three sets of test inputs of varying shapes.
        # Feel free to adjust shapes to match your constraints or desired coverage.

        # --- Test case 1 ---
        cls.input_1 = torch.randn(1, 8, 8)  # (batch_size=1, seq_len=8, d_model=8)
        cls.layer_1 = torch.nn.Sequential(
            HyenaOperator(d_model=8, l_max=8, order=2, filter_order=4)
        )
        # Replace the following with the actual baseline output from your known-good code
        # after running: reference_output_1 = cls.layer_1(cls.input_1)
        # and storing reference_output_1 somewhere safe.
        cls.expected_output_1 = torch.tensor(
            [[[0.1234, -0.5678, 0.1111, 0.2222, 0.3333, -0.4444, 0.5555, -0.6666],
              [0.0000,  0.0000, 0.0000, 0.0000, 0.0000,  0.0000, 0.0000,  0.0000],
              [0.0000,  0.0000, 0.0000, 0.0000, 0.0000,  0.0000, 0.0000,  0.0000],
              [0.0000,  0.0000, 0.0000, 0.0000, 0.0000,  0.0000, 0.0000,  0.0000],
              [0.0000,  0.0000, 0.0000, 0.0000, 0.0000,  0.0000, 0.0000,  0.0000],
              [0.0000,  0.0000, 0.0000, 0.0000, 0.0000,  0.0000, 0.0000,  0.0000],
              [0.0000,  0.0000, 0.0000, 0.0000, 0.0000,  0.0000, 0.0000,  0.0000],
              [0.0000,  0.0000, 0.0000, 0.0000, 0.0000,  0.0000, 0.0000,  0.0000]]]
        )

        # --- Test case 2 ---
        cls.input_2 = torch.randn(2, 10, 8)  # (batch_size=2, seq_len=10, d_model=8)
        cls.layer_2 = torch.nn.Sequential(
            HyenaOperator(d_model=8, l_max=10, order=2, filter_order=8)
        )
        # Replace the following with baseline output from your reference code
        cls.expected_output_2 = torch.tensor(
            [[[-0.7777,  0.8888,  0.1111, -0.2222,  0.3333,  0.4444, -0.5555,  0.6666],
              [ 0.1234, -0.5678,  0.1111,  0.2222,  0.3333, -0.4444,  0.5555, -0.6666],
              [ 0.1234, -0.5678,  0.1111,  0.2222,  0.3333, -0.4444,  0.5555, -0.6666],
              [ 0.1234, -0.5678,  0.1111,  0.2222,  0.3333, -0.4444,  0.5555, -0.6666],
              [ 0.1234, -0.5678,  0.1111,  0.2222,  0.3333, -0.4444,  0.5555, -0.6666],
              [ 0.1234, -0.5678,  0.1111,  0.2222,  0.3333, -0.4444,  0.5555, -0.6666],
              [ 0.1234, -0.5678,  0.1111,  0.2222,  0.3333, -0.4444,  0.5555, -0.6666],
              [ 0.1234, -0.5678,  0.1111,  0.2222,  0.3333, -0.4444,  0.5555, -0.6666],
              [ 0.1234, -0.5678,  0.1111,  0.2222,  0.3333, -0.4444,  0.5555, -0.6666],
              [ 0.1234, -0.5678,  0.1111,  0.2222,  0.3333, -0.4444,  0.5555, -0.6666]],
             [[ 0.9999,  0.7777,  0.8888,  0.1111, -0.2222,  0.3333,  0.4444, -0.5555],
              [ 0.9999,  0.7777,  0.8888,  0.1111, -0.2222,  0.3333,  0.4444, -0.5555],
              [ 0.9999,  0.7777,  0.8888,  0.1111, -0.2222,  0.3333,  0.4444, -0.5555],
              [ 0.9999,  0.7777,  0.8888,  0.1111, -0.2222,  0.3333,  0.4444, -0.5555],
              [ 0.9999,  0.7777,  0.8888,  0.1111, -0.2222,  0.3333,  0.4444, -0.5555],
              [ 0.9999,  0.7777,  0.8888,  0.1111, -0.2222,  0.3333,  0.4444, -0.5555],
              [ 0.9999,  0.7777,  0.8888,  0.1111, -0.2222,  0.3333,  0.4444, -0.5555],
              [ 0.9999,  0.7777,  0.8888,  0.1111, -0.2222,  0.3333,  0.4444, -0.5555],
              [ 0.9999,  0.7777,  0.8888,  0.1111, -0.2222,  0.3333,  0.4444, -0.5555],
              [ 0.9999,  0.7777,  0.8888,  0.1111, -0.2222,  0.3333,  0.4444, -0.5555]]]
        )

        # --- Test case 3 ---
        cls.input_3 = torch.randn(1, 8, 16)  # (batch_size=1, seq_len=8, d_model=16)
        cls.layer_3 = torch.nn.Sequential(
            HyenaOperator(d_model=16, l_max=8, order=2, filter_order=8)
        )
        # Replace the following with baseline output
        cls.expected_output_3 = torch.tensor(
            [[[1.2345, -0.1111, 0.2222,  0.3333,  -0.4444, 0.5555, 0.6666, -0.7777,
               0.9999, -0.9999, 0.8888, -0.8888,  0.6666, 0.4444, -0.2222, 0.1111],
              [1.2345, -0.1111, 0.2222,  0.3333,  -0.4444, 0.5555, 0.6666, -0.7777,
               0.9999, -0.9999, 0.8888, -0.8888,  0.6666, 0.4444, -0.2222, 0.1111],
              [1.2345, -0.1111, 0.2222,  0.3333,  -0.4444, 0.5555, 0.6666, -0.7777,
               0.9999, -0.9999, 0.8888, -0.8888,  0.6666, 0.4444, -0.2222, 0.1111],
              [1.2345, -0.1111, 0.2222,  0.3333,  -0.4444, 0.5555, 0.6666, -0.7777,
               0.9999, -0.9999, 0.8888, -0.8888,  0.6666, 0.4444, -0.2222, 0.1111],
              [1.2345, -0.1111, 0.2222,  0.3333,  -0.4444, 0.5555, 0.6666, -0.7777,
               0.9999, -0.9999, 0.8888, -0.8888,  0.6666, 0.4444, -0.2222, 0.1111],
              [1.2345, -0.1111, 0.2222,  0.3333,  -0.4444, 0.5555, 0.6666, -0.7777,
               0.9999, -0.9999, 0.8888, -0.8888,  0.6666, 0.4444, -0.2222, 0.1111],
              [1.2345, -0.1111, 0.2222,  0.3333,  -0.4444, 0.5555, 0.6666, -0.7777,
               0.9999, -0.9999, 0.8888, -0.8888,  0.6666, 0.4444, -0.2222, 0.1111],
              [1.2345, -0.1111, 0.2222,  0.3333,  -0.4444, 0.5555, 0.6666, -0.7777,
               0.9999, -0.9999, 0.8888, -0.8888,  0.6666, 0.4444, -0.2222, 0.1111]]]
        )

    def test_hyena_case_1(self):
        """
        Test the HyenaOperator output for a small (1,8,8) input.
        """
        with torch.no_grad():
            output_1 = self.layer_1(self.input_1)
        # Check shape
        self.assertEqual(output_1.shape, self.expected_output_1.shape)
        # Check numerical closeness
        if torch.allclose(output_1, self.expected_output_1, atol=0.1, rtol=0.01):
            logging.info(f"Case 1: Test Passed: Expect {self.expected_output_1}, get {output_1}")
        else:
            logging.info(f"Case 1: Test Failed: Expect {self.expected_output_1}, get {output_1}")

    def test_hyena_case_2(self):
        """
        Test the HyenaOperator output for a small (2,10,8) input.
        """
        with torch.no_grad():
            output_2 = self.layer_2(self.input_2)
        # Check shape
        self.assertEqual(output_2.shape, self.expected_output_2.shape)
        # Check numerical closeness
        if torch.allclose(output_2, self.expected_output_2, atol=0.1, rtol=0.01):
            logging.info(f"Case 2: Test Passed: Expect {self.expected_output_2}, get {output_2}")
        else:
            logging.info(f"Case 2: Test Failed: Expect {self.expected_output_2}, get {output_2}")


    def test_hyena_case_3(self):
        """
        Test the HyenaOperator output for a small (1,8,16) input.
        """
        with torch.no_grad():
            output_3 = self.layer_3(self.input_3)
        # Check shape
        self.assertEqual(output_3.shape, self.expected_output_3.shape)
        # Check numerical closeness
        if torch.allclose(output_3, self.expected_output_3, atol=0.1, rtol=0.01):
            logging.info(f"Case 3: Test Passed: Expect {self.expected_output_3}, get {output_3}")
        else:
            logging.info(f"Case 3: Test Failed: Expect {self.expected_output_3}, get {output_3}")


if __name__ == "__main__":
    unittest.main()