import torch
import torch.nn as nn
from typing import List
import copy, math, os, gc
from contextlib import nullcontext
import utils
from quantize.utils import (
    let_parameters, lwc_parameters, get_omni_parameters, omni_state_dict,
    register_scales_and_zeros, smooth_and_quant_temporary, smooth_and_quant_inplace,
    clear_temp_variable, set_quant_state
)
from models.int_llama_layer import QuantLlamaDecoderLayer
from models.int_opt_layer import QuantOPTDecoderLayer
from models.int_falcon_layer import QuantFalconDecoderLayer
from quantize.int_linear import QuantLinear
from quantize.matryoshka_quantizer import Matryoshka_QuantLinear
try:
    import auto_gptq.nn_modules.qlinear.qlinear_cuda as qlinear_cuda
    import auto_gptq.nn_modules.qlinear.qlinear_triton as qlinear_triton
except:
    print("auto_gptq is required for real quantization")



# =============================================================================
# 학습률 관련 상수
# =============================================================================
LET_LR = 5e-4  # LET 학습률
LWC_LR = 1e-3  # LWC 학습률

# =============================================================================
# 보조 함수들
# =============================================================================
def add_new_module(name, original_module, added_module):
    levels = name.split('.')
    if len(levels) > 1:
        mod_ = original_module
        for l_idx in range(len(levels)-1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], added_module)
    else:
        setattr(original_module, name, added_module)     

def get_named_matquant_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, Matryoshka_QuantLinear)}

def replace_with_matquant_layers(model, args):
    """
    모델의 모든 nn.Linear 레이어를 Matryoshka_QuantLinear로 교체
    """
    if not hasattr(args, 'bit_list'):
        args.bit_list = [8, 4, 2]
    
    if max(args.bit_list) != 8:
        print("경고: 최대 비트가 8이 아닙니다. 연구 요구사항에 따라 8비트로 설정됩니다.")
        args.bit_list = sorted(list(set(args.bit_list + [8])), reverse=True)
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            path = name.split('.')
            parent = model
            for part in path[:-1]:
                if not hasattr(parent, part):
                    continue
                parent = getattr(parent, part)
            if hasattr(parent, path[-1]):
                new_module = Matryoshka_QuantLinear(
                    module, 
                    weight_quant_params=args.weight_quant_params,
                    act_quant_params=args.act_quant_params,
                    bit_list=args.bit_list
                )
                setattr(parent, path[-1], new_module)
    
    return model

# =============================================================================
# omniquant_matquant 함수 (Matryoshka_Quant_Loss만 사용)
# =============================================================================
def omniquant_matquant(
    lm,
    args,
    dataloader,
    act_scales,
    act_shifts,
    logger=None,
):
    """
    MatQuant를 OmniQuant와 통합하여 8비트 모델을 기준으로 나머지 비트들은 슬라이싱으로 얻음.
    
    Args:
        lm (nn.Module): 모델 래퍼
        args: 양자화 매개변수, 비트 리스트, 손실 가중치 등 포함
        dataloader: 보정 데이터 로더
        act_scales, act_shifts: 활성화 스케일/시프트 (OmniQuant 관련)
        logger: 로깅 객체
    Returns:
        양자화된 모델(lm.model)
    """
    logger.info("Matryoshka Quantization 시작...")
    
    # 비트 목록 설정 및 검증
    if not hasattr(args, 'bit_list') or not args.bit_list:
        args.bit_list = [8, 4, 2]
    else:
        if 8 not in args.bit_list:
            args.bit_list = sorted(list(set(args.bit_list + [8])), reverse=True)
            logger.info(f"비트 목록에 8비트 추가됨: {args.bit_list}")
    
    # 모델과 장치 설정
    model = lm.model
    dev = lm.device if hasattr(lm, 'device') else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_cache = model.config.use_cache
    model.config.use_cache = False
    logger.info(f"양자화 장치: {dev}")
    
    # 모델 종류에 따라 디코더 레이어와 관련 파라미터 지정
    is_llama = False
    if "llama" in args.net.lower():
        is_llama = True
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        DecoderLayer = QuantLlamaDecoderLayer
        pairs = {
            "q_proj": "qkv",
            "o_proj": "out",
            "up_proj": "fc1"
        }
        layer_name_prefix = "model.layers"
    elif "opt" in args.net.lower():
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
        DecoderLayer = QuantOPTDecoderLayer
        pairs = {
            "q_proj": "qkv",
            "out_proj": "out",
            "fc1": "fc1"
        }
        layer_name_prefix = "model.decoder.layers"
    elif "falcon" in args.net.lower():
        layers = model.transformer.h
        model.transformer.word_embeddings.to(dev)
        model.transformer.ln_f.to(dev)
        model.lm_head.to(dev)
        DecoderLayer = QuantFalconDecoderLayer
        layer_name_prefix = "model.transformer.h"
    elif 'mixtral' in args.net.lower():
        is_llama = True
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        layer_name_prefix = "model.layers"
    else:
        raise ValueError("현재 opt/llama/Llama-2/falcon/mixtral만 지원합니다")
    
    # 첫 번째 레이어를 장치로 이동
    layers[0] = layers[0].to(dev)
    
    # 데이터 타입 및 캐스팅 설정
    if args.deactive_amp and args.epochs > 0:
        dtype = torch.float
        traincast = nullcontext
    else:
        dtype = torch.float16
        traincast = torch.cuda.amp.autocast
    
    # 입력 텐서 초기화
    inps = torch.zeros(
        (args.nsamples, lm.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0}
    
    # 첫 번째 레이어 입력 캡처 클래스
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.is_llama = False
        
        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs.get("attention_mask", None)
            if self.is_llama:
                cache["position_ids"] = kwargs.get("position_ids", None)
            raise ValueError
    
    # 첫 번째 레이어를 Catcher로 교체
    layers[0] = Catcher(layers[0])
    layers[0].is_llama = is_llama
    
    # 보정 데이터 캡처
    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= args.nsamples:
                break
            try:
                model(batch[0].to(dev))
            except ValueError:
                pass
    
    # 첫 번째 레이어와 임베딩 레이어를 CPU로 이동
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    if "llama" in args.net.lower() or "mixtral" in args.net.lower():
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    elif "opt" in args.net.lower():
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    elif 'falcon' in args.net.lower():
        model.transformer.word_embeddings = model.transformer.word_embeddings.cpu()
    torch.cuda.empty_cache()
    
    # 양자화 및 전체 정밀도 모델을 위한 입력 준비
    quant_inps = inps
    fp_inps = copy.deepcopy(inps)
    fp_inps_2 = copy.deepcopy(inps) if args.aug_loss else None
    
    # 마스크 및 위치 ID 준비
    attention_mask = cache["attention_mask"]
    if attention_mask is not None:
        attention_mask_batch = attention_mask.repeat(args.batch_size, 1, 1, 1) if args.deactive_amp else attention_mask.repeat(args.batch_size, 1, 1, 1).float()
    else:
        logger.info("첫 번째 레이어에서 attention mask가 감지되지 않았습니다")
        attention_mask_batch = None
    
    position_ids = cache["position_ids"] if is_llama else None
    
    # 손실 함수 설정
    loss_func = nn.MSELoss()
    
    # 이전 저장된 파라미터 로드
    if hasattr(args, 'resume') and args.resume:
        omni_parameters = torch.load(args.resume)
    else:
        omni_parameters = {}
    
    # 각 레이어에 대해 양자화 진행
    for i in range(len(layers)):
        logger.info(f"=== 레이어 {i} 양자화 시작 ===")
        layer = layers[i].to(dev)
        
        # 모델 유형에 따라 양자화 레이어 생성
        if "mixtral" in args.net.lower():
            qlayer = copy.deepcopy(layer)
            for name, module in qlayer.named_modules():
                if isinstance(module, nn.Linear) and "gate" not in name:  # 게이트는 양자화하지 않음
                    matquant_linear = Matryoshka_QuantLinear(
                        module, 
                        args.weight_quant_params, 
                        args.act_quant_params,
                        args.bit_list
                    )
                    add_new_module(name, qlayer, matquant_linear)
        else:
            qlayer = DecoderLayer(lm.model.config, layer, args)
            named_linears = {}
            for name, module in qlayer.named_modules():
                if isinstance(module, QuantLinear):
                    named_linears[name] = module
            for name, module in named_linears.items():
                org_module = nn.Linear(module.in_features, module.out_features)
                org_module.weight = nn.Parameter(module.weight.clone())
                org_module.bias = nn.Parameter(module.bias.clone()) if module.bias is not None else None
                matquant_linear = Matryoshka_QuantLinear(
                    org_module, 
                    args.weight_quant_params,
                    args.act_quant_params,
                    args.bit_list
                )
                matquant_linear.set_quant_state(
                    weight_quant=module.use_weight_quant,
                    act_quant=module.use_act_quant
                )
                add_new_module(name, qlayer, matquant_linear)
        
        qlayer = qlayer.to(dev)
        
        # 전체 정밀도 모델 출력 계산
        set_quant_state(qlayer, weight_quant=False, act_quant=False)
        if args.epochs > 0:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    for j in range(args.nsamples):
                        fp_inps[j] = qlayer(fp_inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
                        if args.aug_loss:
                            fp_inps_2[j] = qlayer(quant_inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        
        # 스무딩 파라미터 초기화
        set_quant_state(qlayer, weight_quant=False, act_quant=True)
        qlayer.let = args.let
        use_shift = True
        if is_llama or args.abits == 16:
            use_shift = False  # llama 모델과 weight-only 양자화의 경우 채널별 시프팅 비활성화
        
        if args.let:
            # 채널별 스케일링 및 시프트 초기화
            if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'q_proj'):
                qlayer.register_parameter("qkt_smooth_scale",
                                          nn.Parameter(torch.ones(layer.self_attn.q_proj.out_features, device=dev, dtype=dtype)))
            
            for name, module in qlayer.named_modules():
                if isinstance(module, Matryoshka_QuantLinear):
                    for key in pairs.keys():
                        if key in name:
                            act_key = f"{layer_name_prefix}.{i}.{name}"
                            if act_key in act_scales:
                                act = act_scales[act_key].to(device=dev, dtype=dtype).clamp(min=1e-5)
                                weight = module.weight.abs().max(dim=0)[0].clamp(min=1e-5)
                                scale = (act.pow(args.alpha)/weight.pow(1-args.alpha)).clamp(min=1e-5)
                                if use_shift and not is_llama and act_key in act_shifts:
                                    shift = act_shifts[act_key].to(device=dev, dtype=dtype)
                                else:
                                    shift = torch.zeros_like(scale)
                                qlayer.register_parameter(f"{pairs[key]}_smooth_shift", nn.Parameter(shift))
                                qlayer.register_parameter(f"{pairs[key]}_smooth_scale", nn.Parameter(scale))
        
        # 이전에 저장된 파라미터가 있으면 로드
        if args.resume:
            qlayer.load_state_dict(omni_parameters[i], strict=False)
        
        # 훈련 루프
        if args.epochs > 0:
            # AMP 훈련을 위해 float 형식으로 변환
            with torch.no_grad():
                qlayer.float()
            
            # 옵티마이저 생성
            optimizer = torch.optim.AdamW(
                [{"params": let_parameters(qlayer, use_shift), "lr": args.let_lr},
                 {"params": lwc_parameters(qlayer), "lr": args.lwc_lr}],
                weight_decay=args.wd
            )
            loss_scaler = utils.NativeScalerWithGradNormCount()
            
            for epoch in range(args.epochs):
                loss_list = []
                norm_list = []
                
                for j in range(args.nsamples // args.batch_size):
                    index = j * args.batch_size
                    
                    # 양자화 모델 출력 계산
                    with traincast():
                        smooth_and_quant_temporary(qlayer, args, is_llama)
                        
                        # 모든 비트에 대한 양자화 손실 계산
                        total_loss = 0.0
                        
                        # 기본 양자화 출력 (8비트) 계산
                        quant_out = qlayer(quant_inps[index:index+args.batch_size], 
                                          attention_mask=attention_mask_batch,
                                          position_ids=position_ids)[0]
                        
                        # 기본 MSE 손실 계산
                        base_loss = loss_func(fp_inps[index:index+args.batch_size], quant_out)
                        total_loss += base_loss
                        
                        # 나머지 비트에 대한 손실 계산 (8비트 제외)
                        other_bits = [bit for bit in args.bit_list if bit != 8]
                        
                        for bit in other_bits:
                            # 현재 비트 설정
                            for name, module in qlayer.named_modules():
                                if isinstance(module, Matryoshka_QuantLinear):
                                    module.current_bit = bit
                            
                            # 현재 비트에 대한 출력 계산
                            bit_quant_out = qlayer(quant_inps[index:index+args.batch_size],
                                                  attention_mask=attention_mask_batch,
                                                  position_ids=position_ids)[0]
                            
                            # 해당 비트에 대한 손실 계산 및 총 손실에 추가
                            bit_loss = loss_func(fp_inps[index:index+args.batch_size], bit_quant_out)
                            total_loss += bit_loss
                            logger.info(f"레이어 {i} 에포크 {epoch} 배치 {j} {bit}비트 손실: {bit_loss.item():.6f}")
                        
                        # 기본 비트(8비트)로 복원
                        for name, module in qlayer.named_modules():
                            if isinstance(module, Matryoshka_QuantLinear):
                                module.current_bit = 8
                        
                        # 최종 손실 설정
                        loss = total_loss
                        
                        if args.aug_loss:
                            aug_loss = loss_func(fp_inps_2[index:index+args.batch_size], quant_out)
                            loss += aug_loss
                    
                    if not math.isfinite(loss.item()):
                        logger.info("손실이 NAN, 훈련 중단")
                        break
                    
                    loss_list.append(loss.detach().cpu())
                    optimizer.zero_grad()
                    norm = loss_scaler(loss, optimizer, parameters=get_omni_parameters(qlayer, use_shift)).cpu()
                    norm_list.append(norm.data)
                
                loss_mean = torch.stack(loss_list).mean()
                norm_mean = torch.stack(norm_list).mean()
                logger.info(f"레이어 {i} 에포크 {epoch} 손실:{loss_mean} 노름:{norm_mean} 최대 메모리 사용량 {torch.cuda.max_memory_allocated(dev) / 1024**2}MB")
            
            # 임시 변수 정리
            clear_temp_variable(qlayer)
            del optimizer
        
        # half 정밀도로 변환 및 스무딩과 양자화 적용
        qlayer.half()
        smooth_and_quant_inplace(qlayer, args, is_llama)
        
        # 훈련 후 양자화 모델의 입력 업데이트
        if args.epochs > 0:
            with torch.no_grad():
                with traincast():
                    for j in range(args.nsamples):
                        outputs = qlayer(quant_inps[j].unsqueeze(0),
                                       attention_mask=attention_mask,
                                       position_ids=position_ids)
                        output_tensor = outputs[0] if isinstance(outputs, tuple) else outputs
                        quant_inps[j] = output_tensor
            
            # 스케일과 제로 포인트 등록 및 CPU로 이동
            register_scales_and_zeros(qlayer)
            layers[i] = qlayer.to("cpu")
            omni_parameters[i] = omni_state_dict(qlayer)
            torch.save(omni_parameters, os.path.join(args.output_dir, f"omni_parameters.pth"))
        else:
            register_scales_and_zeros(qlayer)
            layers[i] = qlayer.to("cpu")
        
        # 실제 양자화 (auto_gptq 사용)
        if hasattr(args, 'real_quant') and args.real_quant:
            assert args.wbits in [2, 3, 4] and args.abits >= 16  # 지원되는 비트 수 확인
            named_linears = get_named_matquant_linears(qlayer)
            for name, module in named_linears.items():
                if hasattr(module, 'weight_quantizer'):
                    scales = module.weight_quantizer.scales
                    zeros = module.weight_quantizer.zeros
                    group_size = module.weight_quantizer.group_size
                    dim0 = module.weight.shape[0]
                    scales = scales.view(dim0, -1)
                    zeros = zeros.view(dim0, -1)
                    
                    # 적절한 양자화 함수 선택
                    if args.wbits == 3:
                        q_linear = qlinear_cuda.QuantLinear(args.wbits, group_size, module.in_features, module.out_features, not module.bias is None)
                    else:
                        q_linear = qlinear_triton.QuantLinear(args.wbits, group_size, module.in_features, module.out_features, not module.bias is None)
                    
                    # 양자화 적용
                    q_linear.pack(module.cpu(), scales.float().cpu(), zeros.float().cpu())
                    add_new_module(name, qlayer, q_linear)
                    logger.info(f"패킹된 양자화 {name} 완료")
                    del module
        
        # 메모리 해제
        del layer
        gc.collect()
        torch.cuda.empty_cache()
    
    logger.info("Matryoshka Quantization 완료!")
    
    # 모든 Matryoshka_QuantLinear 모듈을 8비트로 설정
    for name, module in model.named_modules():
        if isinstance(module, Matryoshka_QuantLinear):
            module.current_bit = 8
            if hasattr(module, 'cached_weight'):
                module.cached_weight = None
    logger.info("모든 레이어의 양자화 완료, 모든 Matryoshka_QuantLinear 모듈이 8비트로 설정됨")
    
    # 모델 파라미터 저장
    if hasattr(args, 'save_dir') and args.save_dir:
        # 기본 8비트 모델 저장
        save_dict = omni_state_dict(model)
        torch.save(save_dict, f"{args.save_dir}/quant_dict_8bit.pt")
        
        # 각 비트 폭별로 모델 가중치 슬라이싱 및 저장
        for bit in args.bit_list:
            if bit == 8:
                continue  # 이미 8비트는 저장했으므로 스킵
                
            # 현재 비트로 모든 Matryoshka_QuantLinear 모듈 설정
            for name, module in model.named_modules():
                if isinstance(module, Matryoshka_QuantLinear):
                    module.current_bit = bit
                    
                    # 8비트 가중치가 캐싱되어 있지 않다면 생성
                    if module.cached_weight is None and hasattr(module, 'weight_quantizer'):
                        module.weight_quantizer = module.weight_quantizer.to(module.weight.device)
                        module.cached_weight = module.weight_quantizer(module.weight)
            
            # 현재 비트에 대한 모델 상태 딕셔너리 저장
            bit_save_dict = omni_state_dict(model)
            torch.save(bit_save_dict, f"{args.save_dir}/quant_dict_{bit}bit.pt")
            logger.info(f"{bit}비트 양자화 모델이 {args.save_dir}/quant_dict_{bit}bit.pt에 저장됨")
        
        # 모든 모듈을 다시 8비트로 복원
        for name, module in model.named_modules():
            if isinstance(module, Matryoshka_QuantLinear):
                module.current_bit = 8
        
        # 설정 저장
        torch.save(args, f"{args.save_dir}/quant_args.pt")
        logger.info(f"양자화 파라미터가 {args.save_dir}에 저장됨")
    
    # 메모리 정리
    del inps
    del quant_inps
    del fp_inps
    if fp_inps_2 is not None:
        del fp_inps_2
    torch.cuda.empty_cache()
    gc.collect()
    
    # 사용 캐시 설정 복원
    model.config.use_cache = use_cache
    
    return model