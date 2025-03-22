import numpy as np
import argparse
import os
import torch
import time
import logging
import sys
import json
import random
from pathlib import Path
from lm_eval import evaluator
from pprint import pprint
from tqdm import tqdm
import torch.nn as nn
import gc
from contextlib import nullcontext  # AMP 비활성화시 사용할 nullcontext

# 필요한 모듈 임포트
from models.LMClass import LMClass
from utils import create_logger
from quantize.utils import set_quant_state
from datautils import get_loaders
from parallel_utils import map_layers_to_multi_gpus, get_lowest_occupied_gpu
from quantize.matryoshka_omniquant import omniquant_matquant, replace_with_matquant_layers
from quantize.matryoshka_quantizer import Matryoshka_QuantLinear
from categories import subcategories, categories
torch.backends.cudnn.benchmark = True

# 학습률 관련 상수
LET_LR = 1e-3  # 학습 가능한 등가 변환(LET) 학습률
LWC_LR = 1e-3  # 학습 가능한 가중치 클리핑(LWC) 학습률

# 사용자 정의 예외 클래스
class MatQuantError(Exception):
    """MatQuant 작업 중 발생한 오류를 처리하기 위한 예외 클래스"""
    pass

class DeviceMismatchError(MatQuantError):
    """장치 불일치 오류를 처리하기 위한 예외 클래스"""
    def __init__(self, expected_device, actual_device, module_name):
        self.expected_device = expected_device
        self.actual_device = actual_device
        self.module_name = module_name
        message = f"장치 불일치: {module_name}의 예상 장치 {expected_device}와 실제 장치 {actual_device}가 일치하지 않습니다."
        super().__init__(message)

# 오류 발생 시 상세 정보 확인을 위한 장치 정보 출력 함수
def get_device_map(model):
    """모델의 각 부분이 어떤 장치에 있는지 매핑 정보 반환"""
    device_map = {}
    for name, module in model.named_modules():
        if hasattr(module, 'weight'):
            device_map[name] = module.weight.device
        elif list(module.parameters()):
            params = list(module.parameters())
            if params:
                device_map[name] = params[0].device
    return device_map

# 메모리 정보 출력 함수
def print_gpu_memory_usage():
    """현재 GPU 메모리 사용량을 MB 단위로 출력"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**2
            reserved = torch.cuda.memory_reserved(i) / 1024**2
            print(f"GPU {i}: 할당됨 {allocated:.1f} MB, 예약됨 {reserved:.1f} MB")

net_choices = [
    "opt-125m",
    "opt-1.3b",
    "opt-2.7b",
    "opt-6.7b",
    "opt-13b",
    "opt-30b",
    "opt-66b",
    "llama-7b",
    "llama-13b",
    "llama-30b",
    "llama-65b",
    "Llama-2-7b",
    "Llama-2-13b",
    "Llama-2-70b",
    "Llama-2-7b-chat",
    "Llama-2-13b-chat",
    "llava-llama-2-13b-chat-lightning-preview",
    "falcon-180b",
    "falcon-7b",
    "mixtral-8x7b"
]

@torch.inference_mode()
def remove_quantizer_params(model):
    """
    모델에서 양자화기 관련 불필요한 매개변수를 제거하는 함수
    - 모델 저장 크기를 줄이기 위해 사용
    
    Args:
        model: 처리할 모델
        
    Returns:
        int: 제거된 매개변수 수
    """
    removed_count = 0
    for name, module in model.named_modules():
        if isinstance(module, Matryoshka_QuantLinear):
            # 가중치 양자화기 매개변수 초기화 (저장 불필요)
            if hasattr(module, 'weight_quantizer'):
                # 양자화기의 관련 매개변수 제거
                if hasattr(module.weight_quantizer, 'scale') and module.weight_quantizer.scale is not None:
                    module.weight_quantizer.scale = None
                    removed_count += 1
                if hasattr(module.weight_quantizer, 'zero_point') and module.weight_quantizer.zero_point is not None:
                    module.weight_quantizer.zero_point = None
                    removed_count += 1
                
                # 학습 중 사용된 임시 통계 정보 제거
                if hasattr(module.weight_quantizer, 'running_min'):
                    module.weight_quantizer.running_min = None
                    removed_count += 1
                if hasattr(module.weight_quantizer, 'running_max'):
                    module.weight_quantizer.running_max = None
                    removed_count += 1
            
            # 활성화 양자화기 매개변수 초기화 (저장 불필요)
            if hasattr(module, 'act_quantizer') and module.act_quantizer is not None:
                # 양자화기의 관련 매개변수 제거
                if hasattr(module.act_quantizer, 'scale') and module.act_quantizer.scale is not None:
                    module.act_quantizer.scale = None
                    removed_count += 1
                if hasattr(module.act_quantizer, 'zero_point') and module.act_quantizer.zero_point is not None:
                    module.act_quantizer.zero_point = None
                    removed_count += 1
                
                # 학습 중 사용된 임시 통계 정보 제거
                if hasattr(module.act_quantizer, 'running_min'):
                    module.act_quantizer.running_min = None
                    removed_count += 1
                if hasattr(module.act_quantizer, 'running_max'):
                    module.act_quantizer.running_max = None
                    removed_count += 1
    
    return removed_count

def move_all_norm_layers_to_device(model, device):
    """
    모델의 모든 LayerNorm 관련 모듈을 동일한 장치로 이동하는 함수.
    이는 장치 불일치 오류를 방지하기 위해 필요합니다.
    """
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            try:
                module.to(device)
                count += 1
            except Exception as e:
                print(f"레이어 {name}를 {device}로 이동 중 오류 발생: {e}")
    
    # 모델 타입별로 핵심 정규화 모듈 확인 및 이동
    if hasattr(model, 'model'):
        # OPT
        if hasattr(model.model, 'decoder'):
            if hasattr(model.model.decoder, 'final_layer_norm'):
                model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(device)
                count += 1
            
            # 각 레이어의 정규화 모듈 이동
            if hasattr(model.model.decoder, 'layers'):
                for layer in model.model.decoder.layers:
                    if hasattr(layer, 'self_attn_layer_norm'):
                        layer.self_attn_layer_norm = layer.self_attn_layer_norm.to(device)
                        count += 1
                    if hasattr(layer, 'final_layer_norm'):
                        layer.final_layer_norm = layer.final_layer_norm.to(device)
                        count += 1
        
        # LLaMA/Mixtral
        if hasattr(model.model, 'norm'):
            model.model.norm = model.model.norm.to(device)
            count += 1
        
        if hasattr(model.model, 'layers'):
            for layer in model.model.layers:
                if hasattr(layer, 'input_layernorm'):
                    layer.input_layernorm = layer.input_layernorm.to(device)
                    count += 1
                if hasattr(layer, 'post_attention_layernorm'):
                    layer.post_attention_layernorm = layer.post_attention_layernorm.to(device)
                    count += 1
    
        # Falcon
        if hasattr(model, 'transformer'):
            if hasattr(model.transformer, 'ln_f'):
                model.transformer.ln_f = model.transformer.ln_f.to(device)
                count += 1
            
            if hasattr(model.transformer, 'h'):
                for layer in model.transformer.h:
                    if hasattr(layer, 'input_layernorm'):
                        layer.input_layernorm = layer.input_layernorm.to(device)
                        count += 1
                    if hasattr(layer, 'ln_attn'):
                        layer.ln_attn = layer.ln_attn.to(device)
                        count += 1
                    if hasattr(layer, 'ln_mlp'):
                        layer.ln_mlp = layer.ln_mlp.to(device)
                        count += 1
    
    # lm_head를 동일한 장치로 이동
    if hasattr(model, 'lm_head'):
        model.lm_head = model.lm_head.to(device)
        count += 1
    
    return count

@torch.inference_mode()
def evaluate(lm, args, logger):
    """
    MatQuant 모델 평가 함수 (비트폭 하나씩 순차 평가)
    """
    logger.info("평가 시작...")
    results = {}
    
    # Matryoshka_QuantLinear 레이어가 있는지 확인
    matquant_modules = [m for _, m in lm.model.named_modules() if isinstance(m, Matryoshka_QuantLinear)]
    if not matquant_modules:
        logger.warning("모델에 Matryoshka_QuantLinear 레이어가 없습니다. 레이어 교체를 시작합니다.")
        lm.model = replace_with_matquant_layers(lm.model, args)
        logger.info("모든 선형 레이어가 Matryoshka_QuantLinear로 교체되었습니다.")
    else:
        logger.info(f"모델에 {len(matquant_modules)}개의 Matryoshka_QuantLinear 레이어가 있습니다.")
    
    # 장치 설정
    if hasattr(lm, 'device'):
        device = lm.device
    else:
        device = lm._device if hasattr(lm, '_device') else 'cuda:0'
    logger.info(f"평가 장치: {device}")
    
    # 장치 불일치 문제 해결을 위해 모델을 일괄적으로 대상 장치로 이동
    logger.info(f"모델을 {device}로 모두 이동 중...")
    
    # 모델 전체를 CPU로 먼저 이동 (장치 혼합 상태 초기화)
    lm.model.to('cpu')
    torch.cuda.empty_cache()
    
    # 모델 전체를 대상 장치로 이동
    lm.model.to(device)
    
    # 중요 모듈들이 같은 장치에 있는지 확인
    if "opt" in args.net.lower():
        lm.model.model.decoder.embed_tokens = lm.model.model.decoder.embed_tokens.to(device)
        lm.model.model.decoder.embed_positions = lm.model.model.decoder.embed_positions.to(device)
        lm.model.lm_head = lm.model.lm_head.to(device)
        
        # 각 레이어의 중요 모듈들
        for layer in lm.model.model.decoder.layers:
            if hasattr(layer, 'self_attn_layer_norm'):
                layer.self_attn_layer_norm = layer.self_attn_layer_norm.to(device)
            if hasattr(layer, 'final_layer_norm'):
                layer.final_layer_norm = layer.final_layer_norm.to(device)
            
        # final_layer_norm 확인
        if hasattr(lm.model.model.decoder, 'final_layer_norm'):
            lm.model.model.decoder.final_layer_norm = lm.model.model.decoder.final_layer_norm.to(device)
    
    elif "llama" in args.net.lower() or "mixtral" in args.net.lower():
        lm.model.model.embed_tokens = lm.model.model.embed_tokens.to(device)
        lm.model.model.norm = lm.model.model.norm.to(device)
        lm.model.lm_head = lm.model.lm_head.to(device)
        
        # 각 레이어의 중요 모듈들
        for layer in lm.model.model.layers:
            if hasattr(layer, 'input_layernorm'):
                layer.input_layernorm = layer.input_layernorm.to(device)
            if hasattr(layer, 'post_attention_layernorm'):
                layer.post_attention_layernorm = layer.post_attention_layernorm.to(device)
    
    elif "falcon" in args.net.lower():
        lm.model.transformer.word_embeddings = lm.model.transformer.word_embeddings.to(device)
        lm.model.transformer.ln_f = lm.model.transformer.ln_f.to(device)
        lm.model.lm_head = lm.model.lm_head.to(device)
        
        # 각 레이어의 중요 모듈들
        for layer in lm.model.transformer.h:
            if hasattr(layer, 'input_layernorm'):
                layer.input_layernorm = layer.input_layernorm.to(device)
            if hasattr(layer, 'ln_attn'):
                layer.ln_attn = layer.ln_attn.to(device)
            if hasattr(layer, 'ln_mlp'):
                layer.ln_mlp = layer.ln_mlp.to(device)
    
    # LayerNorm 모듈들을 모두 이동
    moved_count = move_all_norm_layers_to_device(lm.model, device)
    logger.info(f"{moved_count}개의 정규화 레이어를 {device}로 이동했습니다.")
    
    # 모든 Matryoshka_QuantLinear 모듈 확인
    for name, module in lm.model.named_modules():
        if isinstance(module, Matryoshka_QuantLinear):
            if hasattr(module, 'weight') and module.weight.device != device:
                module.weight = module.weight.to(device)
            if hasattr(module, 'bias') and module.bias is not None and module.bias.device != device:
                module.bias = module.bias.to(device)
            if hasattr(module, 'weight_quantizer'):
                module.weight_quantizer = module.weight_quantizer.to(device)
    
    # OPT 모델의 경우 모든 레이어 정규화 모듈을 GPU로 이동
    if "opt" in args.net.lower():
        logger.info("OPT 모델의 모든 레이어 정규화 모듈을 GPU로 이동 중...")
    
    # GPU 멀티로 나눠서 평가 모드 설정
    if args.multigpu:
        if "opt" in args.net.lower():
            map_layers_to_multi_gpus(lm.model.model.decoder.layers)
            input_device = lm.model.model.decoder.layers[0].device
            output_device = lm.model.model.decoder.layers[-1].device
            lm._device = input_device
            assert input_device == output_device
            lm.model.model.decoder.embed_positions.to(input_device)
            lm.model.model.decoder.embed_tokens.to(input_device)
            lm.model.model.decoder.final_layer_norm.to(output_device)
            lm.model.lm_head.to(output_device)
        elif "llama" in args.net.lower() or "mixtral" in args.net.lower():
            map_layers_to_multi_gpus(lm.model.model.layers)
            input_device = lm.model.model.layers[0].device
            output_device = lm.model.model.layers[-1].device
            assert input_device == output_device
            lm._device = input_device
            lm.model.model.embed_tokens.to(input_device)
            lm.model.model.norm.to(output_device)
            lm.model.lm_head.to(output_device)
        elif "falcon" in args.net.lower():
            map_layers_to_multi_gpus(lm.model.transformer.h)
            input_device = lm.model.transformer.h[0].device
            output_device = lm.model.transformer.h[-1].device
            assert input_device == output_device
            lm._device = input_device
            lm.model.transformer.word_embeddings.to(input_device)
            lm.model.transformer.ln_f.to(output_device)
            lm.model.lm_head.to(output_device)
    
    # 각 비트별로 평가 수행
    for bit in args.bit_list:
        logger.info(f"=== {bit}-bit 평가 시작 ===")
        bit_results = {}
        lm.model.eval()
        
        # 모든 Matryoshka_QuantLinear 모듈의 비트 폭 설정
        for name, module in lm.model.named_modules():
            if isinstance(module, Matryoshka_QuantLinear):
                # 양자화 활성화
                module.set_quant_state(weight_quant=True, act_quant=True)
                # 현재 비트 설정 (8비트 가중치에서 슬라이싱)
                module.current_bit = bit
                
                # 8비트 양자화 가중치 초기화 (캐싱 유지)
                if bit == 8 and hasattr(module, 'cached_weight'):
                    module.cached_weight = None
                
                # 모듈이 올바른 장치에 있는지 확인
                if hasattr(module, 'weight') and module.weight.device != device:
                    module.weight = module.weight.to(device)
                if hasattr(module, 'bias') and module.bias is not None and module.bias.device != device:
                    module.bias = module.bias.to(device)
        
        # 필요한 데이터 로드
        if args.eval_ppl:
            for dataset in ["wikitext2", "c4"]:
                cache_testloader = f'{args.cache_dir}/testloader_{args.model_family}_{dataset}_all.cache'
                if os.path.exists(cache_testloader):
                    testloader = torch.load(cache_testloader)
                    logger.info(f"캐시에서 테스트 데이터 로드: {cache_testloader}")
                else:
                    dataloader, testloader = get_loaders(
                        dataset,
                        seed=args.seed,
                        model=args.model,
                        seqlen=lm.seqlen,
                    )
                    torch.save(testloader, cache_testloader)
                
                if "c4" in dataset:
                    testenc = testloader
                else:
                    testenc = testloader.input_ids
                
                # PPL 계산
                nsamples = testenc.numel() // lm.seqlen
                use_cache = lm.model.config.use_cache
                lm.model.config.use_cache = False
                nlls = []
                
                for i in tqdm(range(nsamples)):
                    # 입력 텐서를 올바른 장치로 이동
                    batch = testenc[:, (i * lm.seqlen):((i + 1) * lm.seqlen)].to(device)
                    
                    try:
                        if "opt" in args.net.lower():
                            outputs = lm.model.model.decoder(batch)
                        elif "llama" in args.net.lower() or "mixtral" in args.net.lower():
                            outputs = lm.model.model(batch)
                        elif "falcon" in args.net.lower():
                            outputs = lm.model.transformer(batch)
                        
                        # HuggingFace 출력 형식 처리
                        if hasattr(outputs, 'last_hidden_state'):
                            hidden_states = outputs.last_hidden_state
                        elif isinstance(outputs, tuple) and len(outputs) > 0:
                            hidden_states = outputs[0]
                        else:
                            hidden_states = outputs
                        
                        logits = lm.model.lm_head(hidden_states)
                        shift_logits = logits[:, :-1, :]
                        shift_labels = batch[:, 1:].to(logits.device)  # 같은 장치에 있도록 보장
                        
                        loss_fct = nn.CrossEntropyLoss()
                        loss = loss_fct(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1),
                        )
                        neg_log_likelihood = loss.float() * lm.seqlen
                        nlls.append(neg_log_likelihood)
                        
                    except RuntimeError as e:
                        logger.error(f"오류 발생: {e}")
                        # 디버깅을 위한 변수 검사
                        if "Expected all tensors to be on the same device" in str(e):
                            # 오류 위치에 따라 로깅 변수 조정
                            if 'batch' in locals():
                                logger.error(f"배치 장치: {batch.device}")
                            if 'hidden_states' in locals():
                                logger.error(f"hidden_states 장치: {hidden_states.device}")
                            if 'logits' in locals():
                                logger.error(f"로짓 장치: {logits.device}")
                            logger.error(f"lm_head 장치: {lm.model.lm_head.weight.device}")
                            
                            # OPT 모델의 경우 레이어 정규화 장치 확인
                            if "opt" in args.net.lower():
                                logger.error(f"self_attn_layer_norm 장치: {lm.model.model.decoder.layers[0].self_attn_layer_norm.weight.device}")
                        raise
                        
                    if i == args.limit:
                        break

                if nlls:
                    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * lm.seqlen))
                    logger.info(f'{bit}-bit {dataset} PPL: {ppl.item()}')
                    bit_results[dataset] = ppl.item()
                else:
                    logger.warning(f"{dataset}에 대한 PPL 계산을 실패했습니다.")
                    
                lm.model.config.use_cache = use_cache

        # 태스크 평가
        if args.tasks != "":
            t_results = evaluator.simple_evaluate(
                lm,
                tasks=args.tasks,
                num_fewshot=args.num_fewshot,
                limit=None if args.limit == -1 else args.limit,
            )
            for key, value in t_results.items():
                if key != 'results':
                    bit_results[key] = value
            
            logger.info(f"{bit}-bit 평가 결과:")
            pprint(bit_results)
            
            # MMLU 테스트
            if 'hendrycksTest' in args.tasks:
                all_cors = []
                all_cors_norm = []
                subcat_cors = {subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists}
                cat_cors = {cat: [] for cat in categories}
                cat_cors_norm = {cat: [] for cat in categories}
                
                for key in t_results['results'].keys():
                    if not 'hendrycksTest' in key:
                        continue
                    subject = key.split('-')[-1]
                    cors = t_results['results'][key]['acc']
                    cors_norm = t_results['results'][key]['acc_norm']
                    subcats = subcategories[subject]
                    
                    for subcat in subcats:
                        subcat_cors[subcat].append(cors)
                        for key in categories.keys():
                            if subcat in categories[key]:
                                cat_cors[key].append(cors)
                                cat_cors_norm[key].append(cors_norm)
                        all_cors.append(cors)
                        all_cors_norm.append(cors_norm)
                
                for cat in cat_cors:
                    cat_acc = np.mean(cat_cors[cat])
                    logger.info(f"{bit}-bit 평균 정확도 {cat_acc:.4f} - {cat}")
                weighted_acc = np.mean(all_cors)
                logger.info(f"{bit}-bit 전체 평균 정확도: {weighted_acc:.4f}")
        
        # 메모리 사용량 기록
        max_mem = torch.cuda.max_memory_allocated(device) / 1024**2  # MB
        logger.info(f"{bit}-bit 평가 최대 메모리 사용량: {max_mem:.2f} MB")
        
        # 비트별 결과 저장
        results[f"{bit}bit"] = {
            "perplexity": bit_results.get("wikitext2", None),
            "task_results": {k: v for k, v in bit_results.items() if k != "wikitext2" and k != "c4"},
            "memory_usage_mb": max_mem
        }
        
        # 메모리 정리
        torch.cuda.empty_cache()
        logger.info(f"=== {bit}-bit 평가 완료 ===")

    logger.info("모든 비트폭 평가 완료!")
    return results

@torch.inference_mode()
def prepare_model_for_saving(lm, args, logger):
    """
    모델 저장 전에 필요한 사전 작업을 수행하는 함수
    - 모든 Matryoshka_QuantLinear 모듈의 비트를 8로 설정
    - 8비트 양자화 가중치 계산 및 캐싱
    - 불필요한 파라미터 제거
    - 동일한 장치에 모든 레이어를 이동
    
    Returns:
        tuple: (성공 여부, Matryoshka_QuantLinear 모듈 개수)
    """
    try:
        # 저장 전 Matryoshka_QuantLinear 레이어 개수 로깅
        matquant_count = sum(1 for _, module in lm.model.named_modules() 
                             if isinstance(module, Matryoshka_QuantLinear))
        logger.info(f"모델에 {matquant_count}개의 Matryoshka_QuantLinear 레이어가 있습니다.")
        
        if matquant_count == 0:
            logger.warning("모델에 Matryoshka_QuantLinear 레이어가 없습니다. 저장이 잘못될 수 있습니다.")
            return False, 0
        
        # 현재 장치 확인
        if hasattr(lm, '_device'):
            target_device = lm._device
        else:
            target_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        logger.info(f"대상 장치: {target_device}")
        
        # 모든 LayerNorm 모듈을 대상 장치로 이동
        moved_count = move_all_norm_layers_to_device(lm.model, target_device)
        logger.info(f"{moved_count}개의 정규화 레이어가 {target_device}로 이동됨")
        
        # 8비트 양자화 가중치 계산과 각 비트별 상태 딕셔너리 준비
        success_count = 0
        failure_count = 0
        bit_state_dicts = {bit: {} for bit in args.bit_list}
        
        # 각 Matryoshka_QuantLinear 모듈에 대해 8비트 가중치 계산 및 슬라이싱
        for name, module in lm.model.named_modules():
            if isinstance(module, Matryoshka_QuantLinear):
                try:
                    # 8비트 양자화 가중치 계산
                    if hasattr(module, 'weight_quantizer'):
                        module.weight_quantizer = module.weight_quantizer.to(module.weight.device)
                        # 8비트 가중치 계산 및 캐싱
                        module.cached_weight = module.weight_quantizer(module.weight)
                        
                        # 각 비트별로 슬라이싱하여 상태 딕셔너리에 저장
                        for bit in args.bit_list:
                            if bit == 8:
                                # 8비트는 직접 사용
                                quant_weight = module.cached_weight
                            else:
                                # 4비트 또는 2비트는 슬라이싱
                                quant_weight = module.slice_weight(module.cached_weight, bit)
                            
                            # 각 비트별 상태 딕셔너리에 모듈의 가중치 저장
                            module_path = f"{name}.weight"
                            bit_state_dicts[bit][module_path] = quant_weight.clone()
                            
                            # 바이어스가 있으면 추가
                            if module.bias is not None:
                                bias_path = f"{name}.bias"
                                bit_state_dicts[bit][bias_path] = module.bias.clone()
                        
                        success_count += 1
                except Exception as e:
                    logger.warning(f"모듈 {name}의 가중치 캐싱/슬라이싱 실패: {e}")
                    failure_count += 1
        
        logger.info(f"8비트 가중치 캐싱 결과: 성공 {success_count}, 실패 {failure_count}")
        
        # 불필요한 양자화 파라미터 제거
        removed_count = remove_quantizer_params(lm.model)
        logger.info(f"불필요한 양자화 파라미터 {removed_count}개 제거됨")
        
        # 저장 전 상태 확인
        for i, (name, module) in enumerate([(n, m) for n, m in lm.model.named_modules() 
                                        if isinstance(m, Matryoshka_QuantLinear)]):
            if i < 3:
                logger.info(f"저장 준비된 Matryoshka_QuantLinear 레이어 - {name}: bit_list={module.bit_list}, current_bit={module.current_bit if hasattr(module, 'current_bit') else 8}")
            else:
                break
        
        # 각 비트별 상태 딕셔너리 저장
        save_dir = args.save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        for bit in args.bit_list:
            # 기존 상태 딕셔너리에 비트별 가중치 상태 추가
            bit_dict_path = os.path.join(save_dir, f"quant_dict_{bit}bit.pt")
            torch.save(bit_state_dicts[bit], bit_dict_path)
            logger.info(f"{bit}비트 상태 딕셔너리가 저장되었습니다: {bit_dict_path}")
        
        return True, matquant_count
    except Exception as e:
        logger.error(f"모델 저장 준비 중 오류 발생: {e}")
        return False, 0

def main():
    """
    MatQuant + OmniQuant를 실행하는 메인 함수.
    --mode 인자를 통해 quant(양자화)와 eval(평가)을 분리 처리함.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='', help="모델 이름 또는 경로")
    parser.add_argument("--model_family", type=str, default=None)
    parser.add_argument("--cache_dir", default="quantize_cache", type=str, help="데이터셋 캐시 디렉토리")
    parser.add_argument("--output_dir", default="matquant_output", type=str, help="로그 파일 디렉토리")
    parser.add_argument("--save_dir", default="./quantized_models/opt-125m-matquant", type=str, help="양자화 모델 저장 디렉토리")
    parser.add_argument("--mode", type=str, choices=["quant", "eval"], required=True, help="작업 모드를 선택 (quant 또는 eval)")
    parser.add_argument("--wbits", default=4, type=int, choices=[2, 3, 4, 8, 16], help="가중치 양자화 비트 수")
    parser.add_argument("--abits", default=16, type=int, choices=[4, 8, 16], help="활성화 양자화 비트 수")
    parser.add_argument("--group_size", default=128, type=int, help="그룹 크기")
    parser.add_argument("--bit_list", nargs='+', type=int, default=[8, 4, 2], help='MatQuant 슬라이싱을 위한 비트 목록')
    parser.add_argument("--lambda_r", nargs='+', type=float, default=None, help='각 비트 폭에 대한 손실 가중치')
    parser.add_argument("--let", action="store_true", help="학습 가능한 등가 변환 활성화")
    parser.add_argument("--lwc", action="store_true", help="학습 가능한 가중치 클리핑 활성화")
    parser.add_argument("--ste", action="store_true")
    parser.add_argument("--aug_loss", action="store_true", help="동일 입력에 대한 추가 손실 계산")
    parser.add_argument("--epochs", default=10, type=int, help="훈련 에포크 수")
    parser.add_argument("--let_lr", default=LET_LR, type=float, help="LET 학습률")
    parser.add_argument("--lwc_lr", default=LWC_LR, type=float, help="LWC 학습률")
    parser.add_argument("--nsamples", default=128, type=int, help="보정 데이터 샘플 수")
    parser.add_argument("--alpha", default=0.5, type=float, help="알파 파라미터")
    parser.add_argument("--batch_size", default=8, type=int, help="배치 크기")
    parser.add_argument("--real_quant", action="store_true", help="실제 양자화 적용")
    parser.add_argument("--calib_dataset", default="wikitext2", type=str,
                        choices=["wikitext2", "ptb", "c4", "mix", "pile"],
                        help="보정 데이터 추출 위치")
    parser.add_argument("--seed", default=0, type=int, help="랜덤 시드")
    parser.add_argument("--resume", default=None, type=str, help="이어서 학습할 체크포인트")
    parser.add_argument("--act_scales", default=None, type=str, help="활성화 스케일 파일 경로")
    parser.add_argument("--act_shifts", default=None, type=str, help="활성화 시프트 파일 경로")
    parser.add_argument("--deactive_amp", action="store_true", help="AMP 비활성화")
    parser.add_argument("--eval_ppl", action="store_true", help="퍼플렉시티 평가 실행")
    parser.add_argument("--tasks", default="", help="평가 작업")
    parser.add_argument("--num_fewshot", type=int, default=0, help="few-shot 샘플 수")
    parser.add_argument("--limit", type=int, default=-1, help="평가 제한")
    parser.add_argument("--symmetric", action="store_true", help="대칭 양자화")
    parser.add_argument("--disable_zero_point", action="store_true", help="제로 포인트 없는 양자화")
    parser.add_argument("--a_dynamic_method", type=str, default="per_token", choices=["per_token"])
    parser.add_argument("--w_dynamic_method", type=str, default="per_channel", choices=["per_channel"])
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--multigpu", action="store_true", help="평가 시 여러 GPU에 모델 매핑")
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--per_gpu_memory", default=50, type=int)
    parser.add_argument("--wd", default=0, type=float, help="가중치 감쇠")
    parser.add_argument("--fp16", action="store_true", help="fp16 정밀도 사용")
    parser.add_argument("--attn_implementation", type=str, required=False, default="eager",
                        choices=["eager", "sdpa", "flash_attention_2"],
                        help="모델이 작동하는 어텐션 구현")
    parser.add_argument("--net", type=str, default=None, choices=net_choices)
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    if args.epochs > 0:
        assert args.lwc or args.let, "LWC 또는 LET 플래그가 설정되어야 합니다"
        
    if (args.wbits < 16 and args.wbits >= 8) or (args.abits < 16 and args.abits >= 8):
        args.deactive_amp = True
    
    if args.lambda_r is None:
        args.lambda_r = [1.0] * len(args.bit_list)
    
    assert len(args.lambda_r) == len(args.bit_list), "가중치와 비트 목록의 길이가 일치해야 합니다"
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    # if args.save_dir:
    #     os.makedirs(args.save_dir, exist_ok=True)
    
    output_dir = Path(args.output_dir)
    logger = create_logger(output_dir)
    logger.info(args)
    
    if args.net is None:
        if '/' in args.model:
            args.net = args.model.split('/')[-1]
        else:
            args.net = args.model
    if args.model_family is None:
        args.model_family = args.net.split('-')[0]
    
    args.weight_quant_params = {
        "n_bits": args.wbits,
        "per_channel_axes": [0],
        "symmetric": args.symmetric,
        "dynamic_method": args.w_dynamic_method,
        "group_size": args.group_size,
        "lwc": args.lwc,
        "disable_zero_point": args.disable_zero_point
    }
    args.act_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }
    args.q_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }
    args.k_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }
    args.v_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }
    args.p_quant_params = {
        "n_bits": 16,
        "metric": "fix0to1",
    }
    
    if args.mode == "quant":
        logger.info(f"모델 로드 중: {args.model}")
        lm = LMClass(args)
        lm.seqlen = 2048
        lm.model.eval()
        for param in lm.model.parameters():
            param.requires_grad = False
        
        # *** 중요: 여기서 모델을 Matryoshka_QuantLinear로 교체 ***
        logger.info("일반 Linear 레이어를 Matryoshka_QuantLinear로 교체 중...")
        lm.model = replace_with_matquant_layers(lm.model, args)
        matquant_modules = [m for _, m in lm.model.named_modules() if isinstance(m, Matryoshka_QuantLinear)]
        logger.info(f"모델에 {len(matquant_modules)}개의 Matryoshka_QuantLinear 레이어가 있습니다.")
        
        if args.multigpu:
            gpu_id = get_lowest_occupied_gpu(wait_memory=5000)
            lm._device = f"cuda:{gpu_id}"
            logger.info(f"GPU {gpu_id}에서 양자화 설정")
        else:
            if torch.cuda.is_available():
                lm._device = "cuda:0"
                logger.info("CUDA:0에서 양자화 설정")
            else:
                lm._device = "cpu"
                logger.info("CPU에서 양자화 설정")
        
        if "opt" in args.net.lower():
            lm.model.model.decoder.embed_tokens = lm.model.model.decoder.embed_tokens.to(lm._device)
            lm.model.model.decoder.embed_positions = lm.model.model.decoder.embed_positions.to(lm._device)
        elif "llama" in args.net.lower() or "mixtral" in args.net.lower():
            lm.model.model.embed_tokens = lm.model.model.embed_tokens.to(lm._device)
        elif "falcon" in args.net.lower():
            lm.model.transformer.word_embeddings = lm.model.transformer.word_embeddings.to(lm._device)
        
        if args.act_scales is None:
            args.act_scales = f'./act_scales/{args.net}.pt'
        if args.act_shifts is None:
            args.act_shifts = f'./act_shifts/{args.net}.pt'
        
        logger.info("=== Matryoshka Quantization 시작 ===")
        tick = time.time()
        cache_dataloader = f'{args.cache_dir}/dataloader_{args.model_family}_{args.calib_dataset}_{args.nsamples}.cache'
        if os.path.exists(cache_dataloader):
            dataloader = torch.load(cache_dataloader)
            logger.info(f"캐시에서 보정 데이터 로드: {cache_dataloader}")
            cache_testenc = f'{args.cache_dir}/testenc_{args.model_family}_{args.calib_dataset}_{args.nsamples}.cache'
            if os.path.exists(cache_testenc):
                testenc = torch.load(cache_testenc)
                logger.info(f"캐시에서 테스트 데이터 로드: {cache_testenc}")
            else:
                _, testenc = get_loaders(args.calib_dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=lm.seqlen)
                torch.save(testenc, cache_testenc)
        else:
            dataloader, testenc = get_loaders(args.calib_dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=lm.seqlen)
            torch.save(dataloader, cache_dataloader)
            torch.save(testenc, f'{args.cache_dir}/testenc_{args.model_family}_{args.calib_dataset}_{args.nsamples}.cache')
        
        act_scales = None
        act_shifts = None
        if args.let:
            try:
                act_scales = torch.load(args.act_scales)
                act_shifts = torch.load(args.act_shifts)
                logger.info("활성화 스케일 및 시프트 로드됨")
            except:
                logger.warning(f"활성화 스케일 또는 시프트 로드 실패: {args.act_scales}, {args.act_shifts}")
                if args.let:
                    logger.error("LET가 활성화되었지만 활성화 스케일을 찾을 수 없습니다.")
                    sys.exit(1)
        
        omniquant_matquant(lm, args, dataloader, act_scales, act_shifts, logger)
        logger.info(f"총 소요 시간: {time.time() - tick:.2f}초")

    elif args.mode == "eval":
        logger.info("저장된 모델로 재로딩 시작...")
        
        # 평가 결과를 저장할 딕셔너리 초기화
        results = {}
        
        # 저장된 비트별 모델 파일의 존재 여부 확인
        bit_dicts = {}
        for bit in args.bit_list:
            bit_model_path = os.path.join(args.save_dir, f"quant_dict_{bit}bit.pt")
            if os.path.exists(bit_model_path):
                bit_dicts[bit] = bit_model_path
                logger.info(f"{bit}비트 모델 파일 발견: {bit_model_path}")
        
        # 기본 모델 로드 (8비트)
        if 8 in bit_dicts:
            logger.info(f"8비트 모델 로드 중: {bit_dicts[8]}")
            lm = LMClass.from_pretrained(args.save_dir, args=args)
            
            # 저장된 8비트 모델의 상태 딕셔너리 로드
            state_dict = torch.load(bit_dicts[8])
            lm.model.load_state_dict(state_dict, strict=False)
            logger.info("8비트 모델 상태 딕셔너리가 로드되었습니다.")
        else:
            # 8비트 모델이 없으면 일반 방식으로 로드
            lm = LMClass.from_pretrained(args.save_dir, args=args)
        
        lm.seqlen = 2048
        lm.model.eval()
        for param in lm.model.parameters():
            param.requires_grad = False
        
        # 모델에 Matryoshka_QuantLinear 레이어가 없으면 교체
        if not any(isinstance(module, Matryoshka_QuantLinear) for _, module in lm.model.named_modules()):
            logger.info("로드된 모델에서 Matryoshka_QuantLinear 레이어가 없습니다. 레이어 교체 중...")
            lm.model = replace_with_matquant_layers(lm.model, args)
        
        # 각 비트별로 모델 평가
        for bit in args.bit_list:
            if bit in bit_dicts and bit != 8:  # 8비트는 이미 로드되었으므로 스킵
                logger.info(f"{bit}비트 모델 로드 중: {bit_dicts[bit]}")
                # 저장된 비트별 상태 딕셔너리 로드
                state_dict = torch.load(bit_dicts[bit])
                lm.model.load_state_dict(state_dict, strict=False)
                logger.info(f"{bit}비트 모델 상태 딕셔너리가 로드되었습니다.")
                
                # 장치 불일치 문제 해결을 위해 모델을 CPU로 이동 후 다시 대상 장치로 이동
                device = lm._device if hasattr(lm, '_device') else ('cuda:0' if torch.cuda.is_available() else 'cpu')
                logger.info(f"모델을 {device}로 모두 이동 중...")
                
                # 모델 전체를 CPU로 먼저 이동
                lm.model.to('cpu')
                
                # 모든 모듈과 파라미터가 같은 장치에 있도록 확인
                # 모델 전체를 대상 장치로 이동
                lm.model.to(device)
                
                # 특히 임베딩과 LM 헤드가 올바른 장치에 있는지 확인
                if "opt" in args.net.lower():
                    lm.model.model.decoder.embed_tokens = lm.model.model.decoder.embed_tokens.to(device)
                    lm.model.model.decoder.embed_positions = lm.model.model.decoder.embed_positions.to(device)
                    lm.model.lm_head = lm.model.lm_head.to(device)
                elif "llama" in args.net.lower() or "mixtral" in args.net.lower():
                    lm.model.model.embed_tokens = lm.model.model.embed_tokens.to(device)
                    lm.model.model.norm = lm.model.model.norm.to(device)
                    lm.model.lm_head = lm.model.lm_head.to(device)
                elif "falcon" in args.net.lower():
                    lm.model.transformer.word_embeddings = lm.model.transformer.word_embeddings.to(device)
                    lm.model.transformer.ln_f = lm.model.transformer.ln_f.to(device)
                    lm.model.lm_head = lm.model.lm_head.to(device)
                
                # LayerNorm 레이어들도 올바른 장치로 이동
                moved_count = move_all_norm_layers_to_device(lm.model, device)
                logger.info(f"{moved_count}개의 정규화 레이어를 {device}로 이동했습니다.")
                
                # 현재 비트 평가
                logger.info(f"=== {bit}비트 모델 독립 평가 시작 ===")
                bit_results = evaluate_model_with_bit(lm, args, logger, bit)
                
                # 결과 저장
                results[f"{bit}bit_direct"] = bit_results
                logger.info(f"=== {bit}비트 모델 독립 평가 완료 ===")
    
    else:
        logger.error("잘못된 모드입니다. --mode는 quant 또는 eval 이어야 합니다.")
        sys.exit(1)
    
    # Matryoshka 슬라이싱 방식으로 평가
    results.update(evaluate(lm, args, logger))

    if args.save_dir:
        logger.info(f"모델을 {args.save_dir}에 저장 중...")
        
        # 모델 저장 준비
        prepare_result, matquant_count = prepare_model_for_saving(lm, args, logger)
        if prepare_result:
            logger.info("모델 저장 준비 완료")
        else:
            logger.warning("모델 저장 준비에 문제가 있습니다. 계속 진행합니다.")
        
        # 저장 전 모델을 CPU로 이동 (OOM 방지)
        logger.info("모든 모듈을 CPU로 이동...")
        lm.model.to('cpu')
        torch.cuda.empty_cache()
        
        # 가중치 공유 문제 해결
        if "opt" in args.net.lower():
            # OPT 모델은 lm_head와 embed_tokens가 가중치를 공유합니다
            # 이 경우 safetensors에서 오류가 발생하므로 PyTorch 형식으로 저장
            try:
                # config와 tokenizer는 정상적으로 저장
                lm.model.config.save_pretrained(args.save_dir)
                lm.tokenizer.save_pretrained(args.save_dir)
                
                # 저장 전 모델 상태 기록
                logger.info(f"모델 저장 전 구성 확인: Matryoshka_QuantLinear 사용={matquant_count > 0}")
                
                # 모델 구성에 Matryoshka_QuantLinear 사용 정보 추가
                setattr(lm.model.config, 'uses_matryoshka_quant', True)
                setattr(lm.model.config, 'matryoshka_bit_list', args.bit_list)
                lm.model.config.save_pretrained(args.save_dir)
                
                # 모델 상태 딕셔너리 저장
                state_dict = lm.model.state_dict()
                model_path = os.path.join(args.save_dir, "pytorch_model.bin")
                torch.save(state_dict, model_path)
                logger.info(f"모델이 PyTorch 형식으로 저장되었습니다: {model_path}")
            except Exception as e:
                logger.error(f"모델 저장 중 오류 발생: {e}")
        else:
            # 다른 모델은 기본 save_pretrained 메서드 사용
            try:
                # 저장 전 모델 상태 기록
                logger.info(f"모델 저장 전 구성 확인: Matryoshka_QuantLinear 사용={matquant_count > 0}")
                
                # 모델 구성에 Matryoshka_QuantLinear 사용 정보 추가
                setattr(lm.model.config, 'uses_matryoshka_quant', True)
                setattr(lm.model.config, 'matryoshka_bit_list', args.bit_list)
                
                # 모델, 구성 및 토큰화기 저장
                logger.info(f"모델 저장 시작: {args.save_dir}")
                os.makedirs(args.save_dir, exist_ok=True)
                
                # config와 tokenizer 먼저 저장
                lm.model.config.save_pretrained(args.save_dir)
                lm.tokenizer.save_pretrained(args.save_dir)
                
                # 모델 저장
                model_path = os.path.join(args.save_dir, "pytorch_model.bin")
                state_dict = lm.model.state_dict()
                torch.save(state_dict, model_path)
                logger.info(f"모델이 PyTorch 형식으로 저장되었습니다: {model_path}")
            except Exception as e:
                logger.error(f"모델 저장 중 오류 발생: {e}")
        
        # 저장 후 모델 구조 확인 (안전성 개선)
        logger.info("저장 직후 모델 구조 확인을 위한 로드 시도...")
        try:
            model_path = os.path.join(args.save_dir, "pytorch_model.bin")
            # 파일 존재 확인
            if os.path.exists(model_path):
                # 모델 크기 로깅
                model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
                logger.info(f"저장된 모델 파일 크기: {model_size_mb:.2f} MB")
                
                # config 파일 존재 확인
                config_path = os.path.join(args.save_dir, "config.json")
                if os.path.exists(config_path):
                    logger.info("모델 구성 파일이 성공적으로 저장되었습니다.")
                else:
                    logger.warning("모델 구성 파일이 없습니다: config.json")
                    
                # tokenizer 파일 확인
                tokenizer_path = os.path.join(args.save_dir, "tokenizer_config.json")
                if os.path.exists(tokenizer_path):
                    logger.info("토큰화기 구성 파일이 성공적으로 저장되었습니다.")
                else:
                    logger.warning("토큰화기 구성 파일이 없습니다: tokenizer_config.json")
                    
                logger.info("모델 저장이 성공적으로 완료되었습니다.")
            else:
                logger.error(f"모델 파일이 생성되지 않았습니다: {model_path}")
        except Exception as e:
            logger.error(f"저장 후 모델 구조 확인 중 오류 발생: {e}")

    with open(os.path.join(args.output_dir, 'matquant_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    logger.info(f"최종 모델이 {args.output_dir}에 저장되었습니다.")
    logger.info("Matryoshka Quantization 완료!")

# 새로 추가된 함수 - 특정 비트에 대한 모델 평가
@torch.inference_mode()
def evaluate_model_with_bit(lm, args, logger, bit):
    """
    특정 비트로 비트 딕셔너리를 사용해 로드된 모델 평가
    """
    results = {}
    
    # 장치 설정
    if hasattr(lm, 'device'):
        device = lm.device
    else:
        device = lm._device if hasattr(lm, '_device') else 'cuda:0'
    
    logger.info(f"{bit}비트 모델 평가 장치: {device}")
    
    # 모든 모듈이 같은 장치에 있는지 다시 한번 확인
    for name, module in lm.model.named_modules():
        if hasattr(module, 'weight') and module.weight.device != device:
            logger.warning(f"모듈 {name}의 가중치가 {module.weight.device}에 있습니다. {device}로 이동합니다.")
            module.to(device)
    
    # 필요한 데이터 로드 및 평가
    if args.eval_ppl:
        for dataset in ["wikitext2", "c4"]:
            cache_testloader = f'{args.cache_dir}/testloader_{args.model_family}_{dataset}_all.cache'
            if os.path.exists(cache_testloader):
                testloader = torch.load(cache_testloader)
                logger.info(f"캐시에서 테스트 데이터 로드: {cache_testloader}")
            else:
                dataloader, testloader = get_loaders(
                    dataset,
                    seed=args.seed,
                    model=args.model,
                    seqlen=lm.seqlen,
                )
                torch.save(testloader, cache_testloader)
            
            if "c4" in dataset:
                testenc = testloader
            else:
                testenc = testloader.input_ids
            
            # PPL 계산
            nsamples = testenc.numel() // lm.seqlen
            use_cache = lm.model.config.use_cache
            lm.model.config.use_cache = False
            nlls = []
            
            for i in tqdm(range(nsamples)):
                batch = testenc[:, (i * lm.seqlen):((i + 1) * lm.seqlen)].to(device)
                
                try:
                    if "opt" in args.net.lower():
                        outputs = lm.model.model.decoder(batch)
                    elif "llama" in args.net.lower() or "mixtral" in args.net.lower():
                        outputs = lm.model.model(batch)
                    elif "falcon" in args.net.lower():
                        outputs = lm.model.transformer(batch)
                    
                    if hasattr(outputs, 'last_hidden_state'):
                        hidden_states = outputs.last_hidden_state
                    elif isinstance(outputs, tuple) and len(outputs) > 0:
                        hidden_states = outputs[0]
                    else:
                        hidden_states = outputs
                    
                    logits = lm.model.lm_head(hidden_states)
                    shift_logits = logits[:, :-1, :]
                    shift_labels = batch[:, 1:].to(logits.device)
                    
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                    )
                    neg_log_likelihood = loss.float() * lm.seqlen
                    nlls.append(neg_log_likelihood)
                    
                except RuntimeError as e:
                    logger.error(f"오류 발생: {e}")
                    if "Expected all tensors to be on the same device" in str(e):
                        if 'batch' in locals():
                            logger.error(f"배치 장치: {batch.device}")
                        if 'hidden_states' in locals():
                            logger.error(f"hidden_states 장치: {hidden_states.device}")
                        if 'logits' in locals():
                            logger.error(f"로짓 장치: {logits.device}")
                        logger.error(f"lm_head 장치: {lm.model.lm_head.weight.device}")
                    raise
                
                if i == args.limit:
                    break

            if nlls:
                ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * lm.seqlen))
                logger.info(f'{bit}비트 직접 로드 {dataset} PPL: {ppl.item()}')
                results[dataset] = ppl.item()
            else:
                logger.warning(f"{dataset}에 대한 PPL 계산을 실패했습니다.")
            
            lm.model.config.use_cache = use_cache

    # 태스크 평가
    if args.tasks != "":
        t_results = evaluator.simple_evaluate(
            lm,
            tasks=args.tasks,
            num_fewshot=args.num_fewshot,
            limit=None if args.limit == -1 else args.limit,
        )
        for key, value in t_results.items():
            if key != 'results':
                results[key] = value
        
        # MMLU 테스트
        if 'hendrycksTest' in args.tasks:
            all_cors = []
            all_cors_norm = []
            subcat_cors = {subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists}
            cat_cors = {cat: [] for cat in categories}
            cat_cors_norm = {cat: [] for cat in categories}
            
            for key in t_results['results'].keys():
                if not 'hendrycksTest' in key:
                    continue
                subject = key.split('-')[-1]
                cors = t_results['results'][key]['acc']
                cors_norm = t_results['results'][key]['acc_norm']
                subcats = subcategories[subject]
                
                for subcat in subcats:
                    subcat_cors[subcat].append(cors)
                    for key in categories.keys():
                        if subcat in categories[key]:
                            cat_cors[key].append(cors)
                            cat_cors_norm[key].append(cors_norm)
                    all_cors.append(cors)
                    all_cors_norm.append(cors_norm)
            
            for cat in cat_cors:
                cat_acc = np.mean(cat_cors[cat])
                logger.info(f"{bit}비트 직접 로드 평균 정확도 {cat_acc:.4f} - {cat}")
            weighted_acc = np.mean(all_cors)
            logger.info(f"{bit}비트 직접 로드 전체 평균 정확도: {weighted_acc:.4f}")
    
    # 메모리 사용량 기록
    max_mem = torch.cuda.max_memory_allocated(device) / 1024**2  # MB
    logger.info(f"{bit}비트 직접 로드 평가 최대 메모리 사용량: {max_mem:.2f} MB")
    
    # 결과 저장
    results["memory_usage_mb"] = max_mem
    
    # 메모리 정리
    torch.cuda.empty_cache()
    
    return results

if __name__ == '__main__':
    main()