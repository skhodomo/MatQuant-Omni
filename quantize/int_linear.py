import torch
import torch.nn as nn
import torch.nn.functional as F
from quantize.quantizer import UniformAffineQuantizer






class QuantLinear(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(
        self,
        org_module: nn.Linear,
        weight_quant_params: dict = {},
        act_quant_params: dict = {},
        disable_input_quant=False,
    ):
        super().__init__()
        self.fwd_kwargs = dict()
        self.fwd_func = F.linear
        self.register_buffer('weight',org_module.weight)
        if org_module.bias is not None:
            self.register_buffer('bias',org_module.bias)
        else:
            self.bias = None
        self.in_features = org_module.in_features
        self.out_features = org_module.out_features
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        # initialize quantizer
        self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params,shape=org_module.weight.shape)
        if not disable_input_quant:
            self.act_quantizer = UniformAffineQuantizer(**act_quant_params)
        else:
            self.act_quantizer = None

        self.disable_input_quant = disable_input_quant
        self.use_temporary_parameter = False
    def forward(self, input, bit=8):  # bit 인자 추가
        input_dtype = input.dtype

        if self.use_temporary_parameter and hasattr(self, "temp_weight"):
            weight = self.temp_weight.to(input_dtype)
            bias = self.temp_bias.to(input_dtype) if self.temp_bias is not None else None
        else:
            weight = self.weight.to(input_dtype)
            bias = self.bias.to(input_dtype) if self.bias is not None else None

       
        if self.use_weight_quant:
            quantizer = self.weight_quantizer

            # 🛡️ 안전하게 scale / zero point 가져오기
            scale = getattr(quantizer, "scale", getattr(quantizer, "scales", None))
            zp = getattr(quantizer, "round_zero_point", getattr(quantizer, "zeros", None))

            if scale is None:
                raise ValueError("Quantizer has no scale or scales buffer!")
            
            weight = quantizer.fake_quant(weight, scale, zp, bit=bit)
        if self.use_act_quant and self.act_quantizer is not None:
            input = self.act_quantizer(input,16)

        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
