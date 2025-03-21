import os
import torch
import torch.nn as nn
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

from quantize.matryoshka_quantizer import Matryoshka_QuantLinear, Matryoshka_Quant_Loss

# 로깅 설정
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class Config:
    """MatQuant 설정"""
    def __init__(self):
        # 모델 설정
        self.model_name = "facebook/opt-125m"
        
        # 양자화 설정
        self.bit_list = [8, 4, 2]
        self.lambda_r = [1.0, 0.7, 0.5]
        
        # 양자화 매개변수
        self.weight_quant_params = {
            "n_bits": max(self.bit_list),
            "per_channel_axes": [0],
            "symmetric": True,
            "metric": "minmax",
            "group_size": 128,
        }
        
        self.act_quant_params = {
            "n_bits": 8,
            "per_channel_axes": [],
            "symmetric": False,
            "metric": "minmax",
            "dynamic": True,
            "dynamic_method": "per_token",
        }
        
        # 출력 경로
        self.output_dir = "matquant_output/opt-125m-simple"
        os.makedirs(self.output_dir, exist_ok=True)

def convert_to_matquant(model, config):
    """모델을 MatQuant 버전으로 변환"""
    logger.info(f"모델을 MatQuant로 변환합니다 (비트 폭: {config.bit_list})...")
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # 모듈을 포함하는 부모 모듈 찾기
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            
            if parent_name:
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                
            # MatQuant 레이어로 교체
            matquant_linear = Matryoshka_QuantLinear(
                module,
                weight_quant_params=config.weight_quant_params,
                act_quant_params=config.act_quant_params,
                bit_list=config.bit_list
            )
            
            setattr(parent, child_name, matquant_linear)
    
    # 양자화 상태 설정
    for module in model.modules():
        if hasattr(module, 'set_quant_state'):
            module.set_quant_state(weight_quant=True, act_quant=True)
    
    logger.info("모델 변환 완료!")
    return model

def test_generation(model, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """생성 테스트"""
    logger.info("텍스트 생성 테스트를 시작합니다...")
    
    # 입력 프롬프트
    prompts = [
        "Hello, my name is",
        "The meaning of life is",
        "Artificial intelligence will"
    ]
    
    for prompt in prompts:
        logger.info(f"\n프롬프트: {prompt}")
        
        # 토큰화
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # 생성
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=20,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        
        # 결과 출력
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"생성된 텍스트: {generated_text}")
    
    logger.info("텍스트 생성 테스트 완료!")

def main():
    # 설정 로드
    config = Config()
    logger.info(f"모델: {config.model_name}")
    
    # 장치 설정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"사용 장치: {device}")
    
    # 모델 및 토크나이저 로드
    logger.info(f"모델 '{config.model_name}'을(를) 로드합니다...")
    model = AutoModelForCausalLM.from_pretrained(config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # 원본 모델로 토큰 생성 테스트
    logger.info("원본 모델로 생성 테스트 중...")
    model = model.to(device)
    model.eval()
    test_generation(model, tokenizer, device)
    
    # MatQuant 모델로 변환
    matquant_model = convert_to_matquant(model, config)
    
    # MatQuant 모델로 토큰 생성 테스트
    logger.info("MatQuant 모델로 생성 테스트 중...")
    matquant_model = matquant_model.to(device)
    matquant_model.eval()
    test_generation(matquant_model, tokenizer, device)
    
    logger.info(f"모든 테스트가 완료되었습니다!")

if __name__ == "__main__":
    main() 