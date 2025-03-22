import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
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
    - 전체 정밀도 가중치를 8-비트 양자화 표현으로 변환하여 저장
    - 4비트와 2비트는 8비트 가중치를 bit slicing하여, 실제 저장은 8비트만 수행
    - 각 비트 폭에 대해 선형 연산 수행
    """
    def __init__(
        self,
        org_module: nn.Linear,
        weight_quant_params: dict = {},
        act_quant_params: dict = {},
        bit_list: List[int] = [8, 4, 2],
        disable_input_quant: bool = False,
    ):
        super().__init__()
        self.fwd_kwargs = {}
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
        
        # 비트 목록 설정 (내림차순: 가장 높은 비트부터)
        self.bit_list = sorted(bit_list, reverse=True)
        
        # 최대 비트는 항상 8로 고정 (연구 요구사항)
        self.max_bit = 8
        
        # 가중치 양자화기 초기화 (8비트 고정)
        weight_quant_params_copy = weight_quant_params.copy()
        weight_quant_params_copy['n_bits'] = self.max_bit
        self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params_copy, shape=org_module.weight.shape)
        
        # 활성화 양자화기 초기화 (비활성화 옵션에 따라)
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
        S(q_8, r) = clamp(floor(q_8 / 2^(8-r)), 0, 2^r - 1) * 2^(8-r)
        
        8비트 가중치에서 r비트 가중치로 슬라이싱
        """
        c = self.max_bit  # 항상 8
        # 슬라이싱 연산
        sliced = torch.floor(q_weight / (2 ** (c - r)))
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
        전방 통과 - 현재 비트에 맞는 가중치로 선형 변환 계산
        8비트 가중치만 저장하고 나머지는 슬라이싱
        """
        dtype = input.dtype
        device = input.device

        # 임시 파라미터 사용시 (OmniQuant와 호환)
        if self.use_temporary_parameter:
            weight = self.temp_weight.to(device=device, dtype=dtype)
            bias = self.temp_bias.to(device=device, dtype=dtype) if self.temp_bias is not None else None
            return self.fwd_func(input, weight, bias, **self.fwd_kwargs)

        # 가중치 양자화 활성화 상태
        if self.use_weight_quant:
            if hasattr(self, "weight_quantizer"):
                # 가중치 양자화기를 가중치와 동일한 장치에 배치하여 장치 불일치 오류 방지
                weight_device = self.weight.device
                self.weight_quantizer = self.weight_quantizer.to(device=weight_device)

            # 8비트 양자화 가중치 계산 (캐시 여부 확인)
            if self.cached_weight is None:
                try:
                    self.cached_weight = self.weight_quantizer(self.weight)
                except Exception as e:
                    # 오류 발생 시 디버그 정보
                    if hasattr(self, 'weight'):
                        weight_device = self.weight.device
                    else:
                        weight_device = 'unknown'
                    
                    if hasattr(self, 'weight_quantizer'):
                        quantizer_device = next(self.weight_quantizer.parameters(), torch.tensor(0)).device
                    else:
                        quantizer_device = 'unknown'
                    
                    print(f"양자화 오류: 가중치 장치 {weight_device}, 양자화기 장치 {quantizer_device}")
                    # 응급 대응: 장치 불일치 시 해결
                    if weight_device != quantizer_device:
                        self.weight_quantizer = self.weight_quantizer.to(weight_device)
                        self.cached_weight = self.weight_quantizer(self.weight)
                    else:
                        raise

            # 현재 비트가 설정되어 있지 않으면 기본값 사용 (최대 비트)
            bit = self.current_bit if hasattr(self, 'current_bit') else self.bit_list[0]

            # 8비트 가중치 사용 또는 bit slicing 수행
            if bit == 8:
                # 8비트는 직접 사용
                quant_weight = self.cached_weight
            else:
                # 4비트 또는 2비트는 8비트에서 슬라이싱
                quant_weight = self.slice_weight(self.cached_weight, bit)
            
            # 계산용 형식으로 변환
            quant_weight = quant_weight.to(device=device, dtype=dtype)

            # 활성화 양자화 적용
            if self.use_act_quant and not self.disable_input_quant:
                # 활성화 양자화기도 적절한 장치에 있는지 확인
                if hasattr(self, 'act_quantizer') and self.act_quantizer is not None:
                    self.act_quantizer = self.act_quantizer.to(device=device)
                input = self.act_quantizer(input)

            # 선형 연산 수행
            bias = self.bias.to(device=device, dtype=dtype) if self.bias is not None else None
            return self.fwd_func(input, quant_weight, bias, **self.fwd_kwargs)

        # 양자화 비활성화 상태: 일반 선형 레이어처럼 작동
        else:
            weight = self.weight.to(device=device, dtype=dtype)
            bias = self.bias.to(device=device, dtype=dtype) if self.bias is not None else None

            if self.use_act_quant and not self.disable_input_quant:
                # 활성화 양자화기도 적절한 장치에 있는지 확인
                if hasattr(self, 'act_quantizer') and self.act_quantizer is not None:
                    self.act_quantizer = self.act_quantizer.to(device=device)
                input = self.act_quantizer(input)

            return self.fwd_func(input, weight, bias, **self.fwd_kwargs)
    
    def register_scales_and_zeros(self):
        """
        스케일과 제로 포인트 등록 (OmniQuant와 호환)
        """
        self.weight_quantizer.register_scales_and_zeros()
        
    def clear_temp_variable(self):
        """
        임시 변수 제거 (OmniQuant와 호환)
        """
        if hasattr(self, 'temp_weight'):
            del self.temp_weight
        if hasattr(self, 'temp_bias'):
            del self.temp_bias
        self.use_temporary_parameter = False
        
# =============================================================================
class Matryoshka_Quant_Loss(nn.Module):
    """
    다중 비트 손실 함수:
    - 각 비트 폭에 대해 별도로 손실 계산
    - 가중치 lambda_r을 사용하여 집계
    """
    def __init__(self, loss_func=nn.MSELoss(), lambda_r=None, bit_list: List[int] = [8, 4, 2]):
        super().__init__()
        self.loss_func = loss_func
        self.lambda_r = lambda_r if lambda_r else [1.0] * len(bit_list)
        assert len(self.lambda_r) == len(bit_list), "가중치와 비트 목록의 길이가 일치해야 합니다"
        
    def forward(self, quant_outputs, full_precision_output):
        """
        전방 통과 - 여러 비트 폭에 대한 손실 계산 및 집계
        quant_outputs: 리스트, 각 원소는 (batch, ...) 형태의 출력 (비트별)
        full_precision_output: 기준 출력 (full precision)
        """
        total_loss = 0.0
        for i, quant_out in enumerate(quant_outputs):
            loss = self.loss_func(quant_out, full_precision_output)
            total_loss += self.lambda_r[i] * loss
            print(f"loss for bit index {i}: {loss}")
            print("================================================")
        return total_loss 

# 유틸리티 함수 추가
def move_all_norm_layers_to_device(model, device):
    """
    모델 내 모든 정규화 레이어를 특정 장치로 이동시키는 함수
    
    Args:
        model: 처리할 모델
        device: 이동할 대상 장치
    
    Returns:
        int: 이동된 정규화 레이어의 수
    """
    moved_count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            # 이미 원하는 장치에 있는지 확인
            if module.weight.device != device or (module.bias is not None and module.bias.device != device):
                module.to(device)
                moved_count += 1
    
    return moved_count 
