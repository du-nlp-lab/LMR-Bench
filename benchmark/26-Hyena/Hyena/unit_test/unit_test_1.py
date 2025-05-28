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
        self.d_model = d_model
        self.l_max = l_max
        self.order = order
        inner_width = d_model * (order + 1)
        self.dropout = nn.Dropout(dropout)
        self.in_proj = nn.Linear(d_model, inner_width)
        self.out_proj = nn.Linear(d_model, d_model)

        self.short_filter = nn.Conv1d(
            inner_width,
            inner_width,
            3,
            padding=2,
            groups=inner_width
        )
        self.filter_fn = HyenaFilter(
            d_model * (order - 1),
            order=filter_order,
            seq_len=l_max,
            channels=1,
            dropout=filter_dropout,
            **filter_args
        )

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
        cls.expected_output_1 = torch.tensor([[[-0.2499, 0.3312, 0.1442, 0.1878, -0.2253, 0.0747, -0.1653,
                                                0.2745],
                                               [-0.2404, 0.3150, 0.1500, 0.1757, -0.2260, 0.0932, -0.1479,
                                                0.2817],
                                               [-0.2883, 0.3044, 0.0776, 0.0847, -0.2294, 0.1464, -0.1972,
                                                0.3776],
                                               [-0.2913, 0.3046, 0.1412, 0.1694, -0.2236, 0.0646, -0.1867,
                                                0.3253],
                                               [-0.2885, 0.3103, 0.0917, 0.1154, -0.2709, 0.1069, -0.1779,
                                                0.3252],
                                               [-0.3276, 0.3210, 0.0772, 0.1194, -0.2661, 0.0630, -0.2181,
                                                0.3827],
                                               [-0.1796, 0.3299, 0.1579, 0.1776, -0.2431, 0.1432, -0.0890,
                                                0.2354],
                                               [-0.2529, 0.3083, 0.1791, 0.1802, -0.2290, 0.0756, -0.1331,
                                                0.2707]]])

        # --- Test case 2 ---
        cls.input_2 = torch.randn(2, 10, 8)  # (batch_size=2, seq_len=10, d_model=8)
        cls.layer_2 = torch.nn.Sequential(
            HyenaOperator(d_model=8, l_max=10, order=2, filter_order=8)
        )
        # Replace the following with baseline output from your reference code
        cls.expected_output_2 = torch.tensor(
            [[[0.0809, -0.0955, 0.2319, 0.0376, 0.0843, -0.1453, -0.0165, -0.0529],
              [0.3062, -0.1737, 0.0986, -0.1702, 0.0331, 0.0181, -0.1812,
               0.1001],
              [0.1951, 0.0072, 0.1360, 0.1180, 0.1956, -0.1114, -0.1592,
               -0.1342],
              [0.1005, -0.0995, 0.1823, 0.1348, 0.2055, -0.0987, -0.1333,
               -0.1475],
              [0.0059, -0.2544, 0.2801, 0.0918, 0.1486, -0.1635, -0.1793,
               0.0642],
              [0.0790, -0.0551, 0.2446, 0.0449, 0.0596, -0.1387, -0.0734,
               -0.0215],
              [0.0408, -0.0137, 0.2762, 0.1893, 0.1879, -0.2374, -0.1393,
               -0.0611],
              [0.0940, -0.0582, 0.2164, 0.0418, 0.0712, -0.1100, -0.0448,
               -0.0744],
              [0.0786, -0.1946, 0.2228, 0.0314, 0.1227, -0.0965, -0.0879,
               -0.0336],
              [0.1696, -0.1262, 0.1877, 0.0278, 0.1740, -0.1986, -0.0343,
               -0.0240]],

             [[0.0703, -0.1012, 0.2377, 0.0304, 0.0730, -0.1257, -0.0132,
               -0.0614],
              [0.0315, -0.0470, 0.2598, 0.0754, 0.0682, -0.1717, 0.0366,
               -0.0979],
              [0.1423, -0.0851, 0.1971, 0.0410, 0.1462, -0.1133, -0.0623,
               -0.0926],
              [-0.0222, -0.0226, 0.3105, 0.1018, 0.0234, -0.2433, 0.0235,
               -0.0165],
              [0.1255, -0.0490, 0.1885, 0.0577, 0.1061, -0.0856, -0.0949,
               -0.0957],
              [0.0740, -0.0957, 0.2322, 0.0398, 0.0867, -0.1294, -0.0038,
               -0.0806],
              [-0.1761, 0.1410, 0.3809, 0.2605, -0.0470, -0.3592, 0.0487,
               -0.0408],
              [0.0857, -0.0624, 0.2428, 0.0158, 0.0663, -0.0457, -0.0493,
               -0.1119],
              [0.0782, -0.1035, 0.2314, 0.0280, 0.0835, -0.1282, 0.0044,
               -0.0759],
              [0.0300, -0.0980, 0.2596, 0.0650, 0.0889, -0.1684, 0.0325,
               -0.0872]]])

        # --- Test case 3 ---
        cls.input_3 = torch.randn(1, 8, 16)  # (batch_size=1, seq_len=8, d_model=16)
        cls.layer_3 = torch.nn.Sequential(
            HyenaOperator(d_model=16, l_max=8, order=2, filter_order=8)
        )
        # Replace the following with baseline output
        cls.expected_output_3 = torch.tensor([[[-0.1549, 0.0332, 0.2246, 0.1926, -0.2730, -0.0134, -0.0147,
                                                -0.1307, -0.1494, -0.1739, 0.1844, -0.0989, -0.0353, 0.0458,
                                                -0.1252, 0.0141],
                                               [-0.2219, -0.0383, 0.2317, 0.1663, -0.2268, -0.0259, 0.0227,
                                                -0.0160, -0.2080, -0.2005, 0.1757, -0.1025, 0.0075, 0.1009,
                                                -0.1452, 0.0046],
                                               [-0.1203, 0.0892, 0.2451, 0.2053, -0.3023, -0.0226, -0.0671,
                                                -0.1603, -0.0900, -0.1672, 0.1443, -0.0611, -0.0875, 0.0460,
                                                -0.1362, -0.0067],
                                               [-0.1740, -0.0039, 0.2111, 0.2088, -0.3108, -0.0234, -0.0788,
                                                -0.1572, -0.2184, -0.1010, 0.2155, -0.0663, -0.0809, 0.0945,
                                                -0.0715, -0.0037],
                                               [-0.1160, 0.0586, 0.2707, 0.2273, -0.2680, -0.0286, -0.0643,
                                                -0.0984, -0.1716, -0.2345, 0.1939, -0.1808, 0.0034, -0.0295,
                                                -0.1039, 0.0421],
                                               [-0.2800, -0.0561, 0.0844, 0.1213, -0.3365, 0.0523, 0.1276,
                                                -0.1508, -0.1407, -0.1317, 0.3185, -0.1021, -0.1021, 0.1045,
                                                0.0012, -0.0519],
                                               [-0.2750, -0.0210, 0.1329, 0.1084, -0.2885, 0.0235, 0.1142,
                                                -0.0893, -0.1447, -0.1502, 0.2547, -0.0622, -0.0473, 0.1419,
                                                -0.0939, -0.0261],
                                               [-0.1095, 0.0889, 0.2462, 0.1969, -0.2852, -0.0067, -0.0744,
                                                -0.1521, -0.1317, -0.1579, 0.1347, -0.0997, -0.0239, 0.0192,
                                                -0.1586, 0.0027]]])

    def test_hyena_case_1(self):
        """
        Test the HyenaOperator output for a small (1,8,8) input.
        """
        with torch.no_grad():
            output_1 = self.layer_1(self.input_1)
        # Check shape
        self.assertEqual(output_1.shape, self.expected_output_1.shape)
        # Check numerical closeness
        self.assertTrue(
            torch.allclose(output_1, self.expected_output_1, atol=1e-4, rtol=1e-3),
            f"Case 1: output differs from expected by more than tol"
        )

    def test_hyena_case_2(self):
        """
        Test the HyenaOperator output for a small (2,10,8) input.
        """
        with torch.no_grad():
            output_2 = self.layer_2(self.input_2)
        # Check shape
        self.assertEqual(output_2.shape, self.expected_output_2.shape)
        # Check numerical closeness
        self.assertTrue(
            torch.allclose(output_2, self.expected_output_2, atol=1e-4, rtol=1e-3),
            f"Case 2: output differs from expected by more than tol"
        )

    def test_hyena_case_3(self):
        """
        Test the HyenaOperator output for a small (1,8,16) input.
        """
        with torch.no_grad():
            output_3 = self.layer_3(self.input_3)
        # Check shape
        self.assertEqual(output_3.shape, self.expected_output_3.shape)
        # Check numerical closeness
        self.assertTrue(
            torch.allclose(output_3, self.expected_output_3, atol=1e-4, rtol=1e-3),
            f"Case 3: output differs from expected by more than tol"
        )


if __name__ == "__main__":
    unittest.main()
