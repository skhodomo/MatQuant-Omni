import os
import sys
import torch
import torch.nn as nn
import logging

# 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantize.matryoshka_quantizer import Matryoshka_QuantLinear, Matryoshka_Quant_Loss

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(512, 512)
        self.linear2 = nn.Linear(512, 512)
        self.act = nn.GELU()
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x

def convert_to_matquant(model, bit_list=[8, 4, 2]):
    """간단한 모델을 MatQuant 버전으로 변환"""
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # 양자화 매개변수 설정
            weight_quant_params = {
                "n_bits": max(bit_list),
                "per_channel_axes": [0],
                "symmetric": True,
                "metric": "minmax",
                "group_size": 128,  # 그룹 크기 (선택 사항)
            }
            
            act_quant_params = {
                "n_bits": 8,
                "per_channel_axes": [],
                "symmetric": False,
                "metric": "minmax",
                "dynamic": True,
                "dynamic_method": "per_token",
            }
            
            # MatQuant 레이어로 교체
            matquant_linear = Matryoshka_QuantLinear(
                module,
                weight_quant_params=weight_quant_params,
                act_quant_params=act_quant_params,
                bit_list=bit_list
            )
            
            setattr(model, name, matquant_linear)
        elif len(list(module.children())) > 0:
            # 재귀적으로 변환
            convert_to_matquant(module, bit_list)
    
    return model

def main():
    # 더미 모델 생성
    model = SimpleModel()
    logger.info("원본 모델 생성됨")
    
    # 더미 데이터 생성
    dummy_x = torch.randn(32, 512)
    dummy_y = torch.randn(32, 512)
    
    # 원본 모델 출력
    original_output = model(dummy_x)
    logger.info(f"원본 모델 출력 형상: {original_output.shape}")
    
    # MatQuant 모델로 변환
    bit_list = [8, 4, 2]
    matquant_model = convert_to_matquant(model, bit_list)
    logger.info(f"모델이 MatQuant로 변환됨 (비트 폭: {bit_list})")
    
    # 양자화 상태 설정
    for module in matquant_model.modules():
        if hasattr(module, 'set_quant_state'):
            module.set_quant_state(weight_quant=True, act_quant=True)
    
    # MatQuant 모델 출력
    matquant_model.train()  # 훈련 모드로 설정하여 다중 출력 리스트 받기
    matquant_outputs = matquant_model(dummy_x)
    
    logger.info(f"MatQuant 모델 출력 수: {len(matquant_outputs)}")
    for i, output in enumerate(matquant_outputs):
        logger.info(f"{bit_list[i]}-비트 출력 형상: {output.shape}")
    
    # 손실 계산 예제
    loss_fn = Matryoshka_Quant_Loss(
        loss_func=nn.MSELoss(),
        lambda_r=[1.0, 0.7, 0.5],  # 각 비트 폭에 대한 가중치
        bit_list=bit_list
    )
    
    # 손실 계산
    loss = loss_fn(matquant_outputs, original_output)
    logger.info(f"다중 비트 손실: {loss.item()}")
    
    # 추론 모드 시연
    matquant_model.eval()  # 평가 모드로 설정하여 단일 출력 받기
    inference_output = matquant_model(dummy_x[0:1])  # 배치 크기 1
    logger.info(f"추론 모드 출력 형상: {inference_output.shape}")
    
    logger.info("MatQuant 예제가 성공적으로 실행되었습니다!")

if __name__ == "__main__":
    main() 