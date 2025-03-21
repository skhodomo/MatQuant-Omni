import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
from quantize.quantizer import UniformAffineQuantizer
import math

def round_ste(x: torch.Tensor):
    """
    라운딩 연산을 위한 Straight-Through Estimator 구현
    """
    return (x.round() - x).detach() + x

class Matryoshka_QuantLinear(nn.Module):
    """
    Matryoshka 양자화 선형 레이어:
    - 전체 정밀도 가중치를 `c`-비트 양자화 표현으로 변환
    - 여러 `r`-비트 버전으로 조각냄
    - 각 비트 폭에 대해 선형 연산 수행
    """
    def __init__(
        self,
        org_module: nn.Linear,
        weight_quant_params: dict = {},
        act_quant_params: dict = {},
        bit_list: List[int] = [8, 4, 2],
        disable_input_quant=False,
    ):
        super().__init__()
        self.fwd_kwargs = dict()
        self.fwd_func = F.linear
        self.register_buffer('weight', org_module.weight)
        if org_module.bias is not None:
            self.register_buffer('bias', org_module.bias)
        else:
            self.bias = None
        
        self.in_features = org_module.in_features
        self.out_features = org_module.out_features
        
        # 양자화 상태 플래그 (기본적으로 비활성화)
        self.use_weight_quant = False
        self.use_act_quant = False
        
        # 비트 목록 설정
        self.bit_list = sorted(bit_list, reverse=True)  # 가장 높은 비트 폭 먼저
        
        # 최대 비트 폭으로 가중치 양자화기 초기화
        weight_quant_params_copy = weight_quant_params.copy()
        self.max_bit = max(self.bit_list)
        weight_quant_params_copy['n_bits'] = self.max_bit
        self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params_copy, shape=org_module.weight.shape)
        
        # 활성화 양자화기 초기화
        if not disable_input_quant:
            self.act_quantizer = UniformAffineQuantizer(**act_quant_params)
        else:
            self.act_quantizer = None
        
        self.disable_input_quant = disable_input_quant
        self.use_temporary_parameter = False
        self.cached_weight = None
        
    def slice_weight(self, q_weight, r):
        """
        Matryoshka 슬라이싱 연산자 구현:
        S(q_c, r) = clamp(floor(q_c / 2^(c-r)), 0, 2^r - 1) * 2^(c-r)
        """
        c = self.max_bit
        sliced = torch.round(q_weight / (2 ** (c - r)))
        sliced = torch.clamp(sliced, 0, (2 ** r) - 1)
        return sliced * (2 ** (c - r))
    
    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        """
        양자화 상태 설정
        """
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
    
    def forward(self, input: torch.Tensor):
        """
        전방 통과 - 여러 비트 폭으로 선형 변환 계산
        """
        # 임시 파라미터 사용시 (OmniQuant와 호환성 유지)
        if self.use_temporary_parameter:
            weight = self.temp_weight
            bias = self.temp_bias
            output = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
            return output
        
        # 가중치 양자화 활성화 상태
        if self.use_weight_quant:
            # 최대 비트 폭으로 가중치 양자화
            if self.cached_weight is None:
                self.cached_weight = self.weight_quantizer(self.weight)
            
            if self.use_act_quant and not self.disable_input_quant:
                input = self.act_quantizer(input)
                
            # 평가 모드에서는 최고 비트만 사용
            if not self.training:
                # r-비트 가중치로 슬라이싱 (최고 비트)
                r = self.bit_list[0]
                sliced_weight = self.slice_weight(self.cached_weight.clone(), r)
                # 선형 변환 계산
                return self.fwd_func(input, sliced_weight, self.bias, **self.fwd_kwargs)
            
            # 훈련 모드에서는 모든 비트 폭 출력 계산
            outputs = []
            for r in self.bit_list:
                # r-비트 가중치로 슬라이싱
                sliced_weight = self.slice_weight(self.cached_weight.clone(), r)
                # 선형 변환 계산
                outputs.append(self.fwd_func(input, sliced_weight, self.bias, **self.fwd_kwargs))
            
            # 훈련 모드에서는 리스트 대신 첫 번째 출력만 반환 (QuantLinear와 호환성 유지)
            return outputs[0]
        
        # 양자화 비활성화 상태 (일반 선형 레이어처럼 작동)
        else:
            weight = self.weight
            bias = self.bias
            
            if self.use_act_quant and not self.disable_input_quant:
                input = self.act_quantizer(input)
                
            output = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
            return output
    
    def register_scales_and_zeros(self):
        """
        스케일과 제로 포인트 등록 (OmniQuant와 호환성 유지)
        """
        self.weight_quantizer.register_scales_and_zeros()
        
    def clear_temp_variable(self):
        """
        임시 변수 제거 (OmniQuant와 호환성 유지)
        """
        if hasattr(self, 'temp_weight'):
            del self.temp_weight
        if hasattr(self, 'temp_bias'):
            del self.temp_bias
        self.use_temporary_parameter = False
        
class Matryoshka_Quant_Loss(nn.Module):
    """
    다중 비트 손실 함수:
    - 각 비트 폭에 대해 별도로 손실 계산
    - 가중치 lambda_r을 사용하여 집계
    """
    def __init__(self, loss_func=nn.MSELoss(), lambda_r=None, bit_list=[8, 4, 2]):
        super().__init__()
        self.loss_func = loss_func
        self.lambda_r = lambda_r if lambda_r else [1.0] * len(bit_list)
        assert len(self.lambda_r) == len(bit_list), "가중치와 비트 목록의 길이가 일치해야 합니다"
        
    def forward(self, quant_outputs, full_precision_output):
        """
        전방 통과 - 여러 비트 폭에 대한 손실 계산 및 집계
        """
        total_loss = 0.0
        for i, quant_out in enumerate(quant_outputs):
            # 각 비트 폭에 대한 손실 계산
            loss = self.loss_func(quant_out, full_precision_output)
            # 가중치 적용
            total_loss += self.lambda_r[i] * loss
        # 평균화하여 반환
        return total_loss / len(quant_outputs) 