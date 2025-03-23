import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
import tqdm
import numpy as np
import pdb
import math

CLIPMIN = 1e-5




def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x



class UniformAffineQuantizer(nn.Module):
    def __init__(
        self,
        n_bits: int = 8,
        symmetric: bool = False,
        per_channel_axes=[],
        metric="minmax",
        dynamic=False,
        dynamic_method="per_cluster",
        group_size=None,
        shape=None,
        lwc=False,
        disable_zero_point=False,
    ):
        """
        support cluster quantize
        dynamic_method support per_token and per_cluster
        """
        super().__init__()
        self.symmetric = symmetric
        self.disable_zero_point = disable_zero_point
        assert 2 <= n_bits <= 16, "bitwidth not supported"
        self.n_bits = n_bits
        if self.disable_zero_point:
            self.qmin = -(2 ** (n_bits - 1))
            self.qmax = 2 ** (n_bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** (n_bits) - 1
        self.per_channel_axes = per_channel_axes
        self.metric = metric
        self.cluster_counts = None
        self.cluster_dim = None

        self.scale = None
        self.zero_point = None
        self.round_zero_point = None

        self.cached_xmin = None
        self.cached_xmax = None
        self.dynamic = dynamic
        self.dynamic_method = dynamic_method
        self.deficiency = 0
        self.lwc = lwc
        
        init_value = 4.             # inti value of learnable weight clipping
        if lwc:
            if group_size:
                dim1 = int(shape[0]*math.ceil(shape[1]/group_size))
                self.deficiency = shape[-1]%group_size
                if self.deficiency > 0:
                    self.deficiency = group_size - self.deficiency
                    assert self.symmetric   # support for mlc-llm symmetric quantization
            else:
                dim1 = shape[0]
            self.upbound_factor = nn.Parameter(torch.ones((dim1,1))*init_value)
            self.lowbound_factor = nn.Parameter(torch.ones((dim1,1))*init_value)
        self.sigmoid = nn.Sigmoid()

        self.enable = True
        self.group_size = group_size

    def change_n_bits(self, n_bits):
        self.n_bits = n_bits
        if self.disable_zero_point:
            self.qmin = -(2 ** (n_bits - 1))
            self.qmax = 2 ** (n_bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** (n_bits) - 1


    def fake_quant(self, x, scale, round_zero_point, bit):
         # ✅ fallback: scale이 없고 buffer에만 있는 경우
        if scale is None and hasattr(self, "scales"):
            scale = self.scales
        if round_zero_point is None and hasattr(self, "zeros"):
            round_zero_point = self.zeros
        def bit_slice(qc: torch.Tensor, r: int, c: int = 8):
            """
            q^c에서 상위 r비트만 추출하는 Matryoshka bit slicing 연산.

            Args:
                qc (Tensor): 8bit 양자화된 정수 weight (0 ~ 255)
                r (int): 타겟 비트 수 (2, 4, etc)
                c (int): 원래 비트 수 (기본 8)
            
            Returns:
                Tensor: 슬라이스된 정수 weight (0 ~ 2^c - 1 범위)
            """
            shift = 2 ** (c - r)
            sliced = torch.round(qc / shift)       # 상위 r비트 추출
            sliced = torch.clamp(sliced, 0, 2**r - 1)     # r비트 범위로 자름
            sliced_q = sliced * shift                    # 다시 원래 스케일로 복원
            return sliced_q

        if self.deficiency > 0:
            pad_zeros = torch.zeros((x.shape[0],self.deficiency),dtype=x.dtype,device=x.device)
            x = torch.cat((x,pad_zeros),dim=1)
        
        if self.group_size:
            assert len(x.shape)==2, "only support linear layer now"
            dim1, dim2 = x.shape
            x = x.reshape(-1, self.group_size)
        x_int = round_ste(x / scale)
        if round_zero_point is not None:
            x_int = x_int.add(round_zero_point)
        x_int = x_int.clamp(self.qmin, self.qmax)
        x_dequant = x_int
        
        if bit==4:
            x_dequant = bit_slice(x_int,4) 
            if round_zero_point is not None:
                x_dequant = x_dequant.sub(round_zero_point)
            x_dequant = x_dequant.mul(scale)
            if self.group_size:
                x_dequant = x_dequant.reshape(dim1, dim2)
            if self.deficiency > 0:
                x_dequant = x_dequant[:,:-self.deficiency]
            return x_dequant
        if bit==2:
            x_dequant = bit_slice(x_int,4)
            x_dequant = bit_slice(x_dequant,2) 
            if round_zero_point is not None:
                x_dequant = x_dequant.sub(round_zero_point)
            x_dequant = x_dequant.mul(scale)
            if self.group_size:
                x_dequant = x_dequant.reshape(dim1, dim2)
            if self.deficiency > 0:
                x_dequant = x_dequant[:,:-self.deficiency]
        else:
            if round_zero_point is not None:
                x_dequant = x_dequant.sub(round_zero_point)
            x_dequant = x_dequant.mul(scale)
            if self.group_size:
                x_dequant = x_dequant.reshape(dim1, dim2)
            if self.deficiency > 0:
                x_dequant = x_dequant[:,:-self.deficiency]
        return x_dequant
    def forward(self, x: torch.Tensor, bit: int = None):
        if bit is None:
            bit = self.n_bits  # fallback to default

        if bit >= 16 or not self.enable:
            return x

        if self.metric == "fix0to1":
            return x.mul_(2**bit - 1).round_().div_(2**bit - 1)

        if self.dynamic_method in ["per_token", "per_channel"]:
            self.per_token_dynamic_calibration(x)
        else:
            raise NotImplementedError()

        x_dequant = self.fake_quant(x, self.scale, self.round_zero_point, bit)
        return x_dequant
    def per_token_dynamic_calibration(self, x):
        if self.group_size:
            if self.deficiency == 0:
                x = x.reshape(-1,self.group_size)
            else:
                pad_zeros = torch.zeros((x.shape[0],self.deficiency),dtype=x.dtype,device=x.device)
                x = torch.cat((x,pad_zeros),dim=1)
                x = x.reshape(-1,self.group_size)
        reduce_shape = [-1]
        xmin = x.amin(reduce_shape, keepdim=True)
        xmax =  x.amax(reduce_shape, keepdim=True)
        if self.lwc:
            xmax = self.sigmoid(self.upbound_factor)*xmax
            xmin = self.sigmoid(self.lowbound_factor)*xmin
        if self.symmetric:
            abs_max = torch.max(xmax.abs(),xmin.abs())
            scale = abs_max / (2**(self.n_bits-1)-1)
            self.scale = scale.clamp(min=CLIPMIN, max=1e4)
            zero_point = (2**(self.n_bits-1)-1)*torch.ones_like(self.scale)
        else:
            range = xmax - xmin
            scale = range / (2**self.n_bits-1)
            self.scale = scale.clamp(min=CLIPMIN, max=1e4)
            zero_point = -(xmin) / (self.scale)
        if self.disable_zero_point:
            self.round_zero_point = None
        else:
            self.round_zero_point = zero_point.clamp(min=-1e4, max=1e4).round()
        
    def register_scales_and_zeros(self):
        self.register_buffer('scales', self.scale)
        self.register_buffer('zeros', self.round_zero_point)
        del self.scale
        del self.round_zero_point
