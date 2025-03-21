import torch
import torch.nn as nn
from models.int_llama_layer import QuantLlamaDecoderLayer
from models.int_opt_layer import QuantOPTDecoderLayer
from models.int_falcon_layer import QuantFalconDecoderLayer
from quantize.int_linear import QuantLinear
from quantize.matryoshka_quantizer import Matryoshka_QuantLinear, Matryoshka_Quant_Loss
from contextlib import nullcontext
import copy
import math
import utils
import os
import gc
from quantize.utils import let_parameters, lwc_parameters, get_omni_parameters, \
                            omni_state_dict, register_scales_and_zeros, smooth_and_quant_temporary, \
                            smooth_and_quant_inplace, clear_temp_variable, set_quant_state
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
# MatQuant 버전의 named_linears 함수
def get_named_matquant_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, Matryoshka_QuantLinear)}

# MatQuant 레이어로 모델 변환
def replace_with_matquant_layers(model, args):
    """
    모델의 모든 QuantLinear 레이어를 Matryoshka_QuantLinear로 교체
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # 모델 내 선형 레이어 인스턴스 찾기
            path = name.split('.')
            parent = model
            for part in path[:-1]:
                if not hasattr(parent, part):
                    continue
                parent = getattr(parent, part)
            
            # 해당 레이어가 있는 경우에만 교체
            if hasattr(parent, path[-1]):
                new_module = Matryoshka_QuantLinear(
                    module, 
                    weight_quant_params=args.weight_quant_params,
                    act_quant_params=args.act_quant_params,
                    bit_list=args.bit_list
                )
                setattr(parent, path[-1], new_module)
    
    return model

def omniquant_matquant(
    lm,
    args,
    dataloader,
    act_scales,
    act_shifts,
    logger=None,
):
    """
    MatQuant를 OmniQuant에 통합
    
    Args:
        lm (nn.Module): 모델
        args: 양자화 매개변수, 비트 폭, 손실 가중치 등을 포함하는 인수 객체
        dataloader: 보정 데이터 로더
        act_scales: 활성화 스케일 (OmniQuant)
        act_shifts: 활성화 시프트 (OmniQuant)
        logger: 로깅 객체
    
    Returns:
        양자화된 MatQuant 모델
    """
    logger.info("Matryoshka Quantization 시작...")
    
    # OmniQuant 기본 로직과 유사한 코드 구조 유지
    model = lm.model
    
    # dev 변수 설정 개선
    if hasattr(lm, '_device') and lm._device:
        dev = lm._device
    else:
        dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        lm._device = dev
    
    logger.info(f"양자화 장치: {dev}")
    
    # OmniQuant와 동일한 추론 방식 유지
    if "llama" in args.net.lower() or "mixtral" in args.net.lower():
        is_llama = True
        layers = model.layers
        DecoderLayer = QuantLlamaDecoderLayer
        layer_name_prefix = "model.layers"
        pairs = {"q_proj":"qkv", "k_proj":"qkv", "v_proj":"qkv",
                "up_proj":"fc1","gate_proj":"fc1"}
        
    elif "opt" in args.net.lower():
        is_llama = False
        layers = model.model.decoder.layers
        DecoderLayer = QuantOPTDecoderLayer
        layer_name_prefix = "model.decoder.layers"
        pairs = {"q_proj":"qkv", "k_proj":"qkv", "v_proj":"qkv", "fc1":"fc1"}
        
    elif 'falcon' in args.model:
        is_llama = False
        layers = model.h
        DecoderLayer = QuantFalconDecoderLayer
        layer_name_prefix = "model.h"
        pairs = {"query_key_value":"qkv", "dense_h_to_4h":"fc1"}
    else:
        raise ValueError("현재 opt/llama/Llama-2/falcon/mixtral만 지원합니다")
    
    # 손실 함수 초기화
    matquant_loss = Matryoshka_Quant_Loss(
        loss_func=nn.MSELoss(), 
        lambda_r=args.lambda_r if hasattr(args, 'lambda_r') else None,
        bit_list=args.bit_list if hasattr(args, 'bit_list') else [8, 4, 2]
    )
    
    # 캐시 설정
    dtype = torch.float16 if args.fp16 else torch.float32
    inps = torch.zeros(
        (args.nsamples, lm.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    
    # MatQuant 매개변수로 수정
    args_matquant = copy.deepcopy(args)
    if not hasattr(args_matquant, 'bit_list'):
        args_matquant.bit_list = [8, 4, 2]
    if not hasattr(args_matquant, 'lambda_r'):
        args_matquant.lambda_r = [1.0] * len(args_matquant.bit_list)
    
    # 캐시 데이터 준비 (OmniQuant와 동일)
    cache = {"i": 0}
    
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            
        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs.get("attention_mask", None)
            cache["position_ids"] = kwargs.get("position_ids", None)
            raise ValueError
    
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            # 입력 디바이스를 dev로 변경
            batch = batch[0].to(dev)
            
            # 모델 임베딩을 동일한 디바이스로 이동
            if "opt" in args.net.lower():
                model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
                model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
            elif "llama" in args.net.lower() or "mixtral" in args.net.lower():
                model.model.embed_tokens = model.model.embed_tokens.to(dev)
            elif "falcon" in args.net.lower():
                model.transformer.word_embeddings = model.transformer.word_embeddings.to(dev)
                
            lm.model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    
    # 메모리 초기화
    torch.cuda.empty_cache()
    
    # OmniQuant와 유사한 로직으로 각 레이어 처리
    if hasattr(args, 'resume') and args.resume:
        omni_parameters = torch.load(args.resume)
    else:
        omni_parameters = {}
    
    for i in range(len(layers)):
        logger.info(f"=== 레이어 {i} 양자화 시작 ===")
        layer = layers[i].to(dev)
        
        # MatQuant 특화 로직 적용
        if "mixtral" in args.net.lower():
            qlayer = copy.deepcopy(layer)
            for name, module in qlayer.named_modules():
                if isinstance(module, torch.nn.Linear) and not "gate" in name:
                    # Matryoshka_QuantLinear로 교체
                    quantlinear = Matryoshka_QuantLinear(
                        module, 
                        args.weight_quant_params, 
                        args.act_quant_params,
                        args_matquant.bit_list
                    )
                    add_new_module(name, qlayer, quantlinear)
        else:
            # 기존 QuantDecoderLayer 초기화 후 QuantLinear -> Matryoshka_QuantLinear로 변환
            qlayer = DecoderLayer(lm.model.config, layer, args)
            
            # 모든 QuantLinear 모듈 찾기
            named_linears = {}
            for name, module in qlayer.named_modules():
                if isinstance(module, QuantLinear):
                    named_linears[name] = module
            
            # 각 QuantLinear를 Matryoshka_QuantLinear로 변환
            for name, module in named_linears.items():
                # 원본 Linear 매개변수 추출
                org_module = nn.Linear(module.in_features, module.out_features)
                org_module.weight = nn.Parameter(module.weight.clone())
                if module.bias is not None:
                    org_module.bias = nn.Parameter(module.bias.clone())
                else:
                    org_module.bias = None
                
                # Matryoshka_QuantLinear 생성
                matquant_linear = Matryoshka_QuantLinear(
                    org_module, 
                    args.weight_quant_params,
                    args.act_quant_params,
                    args_matquant.bit_list
                )
                
                # set_quant_state 유지
                matquant_linear.set_quant_state(
                    weight_quant=module.use_weight_quant,
                    act_quant=module.use_act_quant
                )
                
                # 모듈 교체
                add_new_module(name, qlayer, matquant_linear)
        
        qlayer = qlayer.to(dev)
        
        # 전체 정밀도 모델 출력 얻기
        set_quant_state(qlayer, weight_quant=False, act_quant=False)
        
        # 입력 데이터 복사
        quant_inps = inps
        fp_inps = copy.deepcopy(inps)
        fp_inps_2 = copy.deepcopy(inps) if args.aug_loss else None
        
        attention_mask = cache["attention_mask"]
        if attention_mask is not None:
            attention_mask_batch = attention_mask.repeat(args.batch_size,1,1,1) if args.deactive_amp else attention_mask.repeat(args.batch_size,1,1,1).float()
        else:
            logger.info("첫 번째 레이어에서 attention mask가 감지되지 않았습니다")
            attention_mask_batch = None
            
        position_ids = cache["position_ids"] if is_llama else None
        
        # 학습 로직
        if args.epochs > 0:
            with torch.no_grad():
                with torch.cuda.amp.autocast() if not args.deactive_amp else nullcontext():
                    for j in range(args.nsamples):
                        out = qlayer(fp_inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)
                        fp_inps[j] = out[0] if isinstance(out, tuple) else out
                        if args.aug_loss:
                            out2 = qlayer(quant_inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)
                            fp_inps_2[j] = out2[0] if isinstance(out2, tuple) else out2
            
            # 양자화 상태 설정
            set_quant_state(qlayer, weight_quant=False, act_quant=True)
            qlayer.let = args.let
            
            use_shift = True
            if is_llama or args.abits == 16:
                use_shift = False
                
            # LET 초기화 (OmniQuant와 동일)
            if args.let:
                try:
                    # 대체 초기화 방법
                    if hasattr(qlayer, "self_attn") and hasattr(qlayer.self_attn, "q_proj"):
                        out_features = qlayer.self_attn.q_proj.out_features
                        qlayer.register_parameter("qkt_smooth_scale", torch.nn.Parameter(torch.ones(out_features, device=dev, dtype=dtype)))
                except Exception as e:
                    logger.warning(f"qkt_smooth_scale 초기화 중 오류 발생: {e}")
                
                # LET 파라미터 초기화
                for name, module in qlayer.named_modules():
                    if isinstance(module, (QuantLinear, Matryoshka_QuantLinear)):
                        for key in pairs.keys():
                            if key in name:
                                try:
                                    act_key = f"{layer_name_prefix}.{i}.{name}"
                                    if act_key in act_scales:
                                        act = act_scales[act_key].to(device=dev, dtype=dtype).clamp(min=1e-5)
                                        weight = module.weight.abs().max(dim=0)[0].clamp(min=1e-5)
                                        scale = (act.pow(args.alpha)/weight.pow(1-args.alpha)).clamp(min=1e-5)
                                        if use_shift and not is_llama:
                                            shift = act_shifts[act_key].to(device=dev, dtype=dtype)
                                        else:
                                            shift = torch.zeros_like(scale)
                                        qlayer.register_parameter(f"{pairs[key]}_smooth_shift", torch.nn.Parameter(shift))
                                        qlayer.register_parameter(f"{pairs[key]}_smooth_scale", torch.nn.Parameter(scale))
                                except Exception as e:
                                    logger.warning(f"LET 초기화 중 오류 발생 ({name}, {key}): {e}")
                                    continue
            
            if args.resume:
                qlayer.load_state_dict(omni_parameters[i], strict=False)
            
            # 최적화기 설정
            with torch.no_grad():
                qlayer.float()
            
            optimizer = torch.optim.AdamW(
                [{"params": let_parameters(qlayer, use_shift), "lr": args.let_lr}, 
                 {"params": lwc_parameters(qlayer), "lr": args.lwc_lr}],
                weight_decay=args.wd
            )
            loss_scaler = utils.NativeScalerWithGradNormCount()
            
            # 학습 루프
            traincast = torch.cuda.amp.autocast if not args.deactive_amp else nullcontext
            for epochs in range(args.epochs):
                loss_list = []
                norm_list = []
                
                for j in range(args.nsamples // args.batch_size):
                    index = j * args.batch_size
                    
                    # 양자화 모델 출력 계산
                    with traincast():
                        smooth_and_quant_temporary(qlayer, args, is_llama)
                        
                        # 모든 Matryoshka_QuantLinear 레이어의 비트 폭 출력을 수집하기 위한 훅 설정
                        bit_outputs = {}
                        hooks = []
                        
                        def hook_fn(module, input, output, name):
                            bit_outputs[name] = []
                            # 각 비트 폭마다 출력 저장
                            if hasattr(module, 'bit_list') and module.cached_weight is not None:
                                for r in module.bit_list:
                                    # r-비트 가중치로 슬라이싱
                                    sliced_weight = module.slice_weight(module.cached_weight.clone(), r)
                                    # 선형 변환 계산
                                    result = module.fwd_func(input[0], sliced_weight, module.bias, **module.fwd_kwargs)
                                    bit_outputs[name].append(result)
                        
                        # 각 Matryoshka_QuantLinear에 후크 등록
                        for name, module in qlayer.named_modules():
                            if isinstance(module, Matryoshka_QuantLinear) and module.use_weight_quant:
                                hooks.append(module.register_forward_hook(
                                    lambda mod, inp, out, nm=name: hook_fn(mod, inp, out, nm)
                                ))
                        
                        # 정상적인 순방향 계산 수행
                        output = qlayer(quant_inps[index:index+args.batch_size], 
                                     attention_mask=attention_mask_batch,
                                     position_ids=position_ids)
                        
                        # 후크 제거
                        for hook in hooks:
                            hook.remove()
                            
                        # 모든 레이어에서 비트 출력 수집
                        if bit_outputs and args_matquant.bit_list:
                            # MSE 손실 계산
                            target = fp_inps[index:index+args.batch_size]
                            # output이 튜플인 경우 첫 번째 요소만 사용
                            if isinstance(output, tuple):
                                model_output = output[0]
                            else:
                                model_output = output
                                
                            # target이 튜플인 경우 첫 번째 요소만 사용
                            if isinstance(target, tuple):
                                target = target[0]
                                
                            base_loss = nn.MSELoss()(model_output, target)
                            
                            # Matryoshka 추가 손실 계산 (비트 폭 당 손실의 가중 평균)
                            bit_loss = 0.0
                            count = 0
                            
                            # 중요한 레이어만 고려 (QKV, FFN)
                            for name, outputs_list in bit_outputs.items():
                                if any(key in name for key in pairs.keys()):
                                    for i, bit_output in enumerate(outputs_list[1:], 1):  # 첫 번째는 이미 base_loss에 포함됨
                                        lambda_r = args_matquant.lambda_r[i] if i < len(args_matquant.lambda_r) else args_matquant.lambda_r[-1]
                                        # 레이어 출력은 사용할 수 없으므로 인라인 손실 계산
                                        bit_loss += lambda_r * torch.nn.functional.mse_loss(bit_output, outputs_list[0].detach())
                                        count += 1
                            
                            # 최종 손실 계산
                            if count > 0:
                                bit_loss = bit_loss / count
                                loss = base_loss + bit_loss * 0.1  # 가중치 조정 가능
                            else:
                                loss = base_loss
                                
                            if args.aug_loss and fp_inps_2 is not None:
                                aug_target = fp_inps_2[index:index+args.batch_size]
                                if isinstance(aug_target, tuple):
                                    aug_target = aug_target[0]
                                loss += nn.MSELoss()(model_output, aug_target)
                        else:
                            # 기존 손실 계산 방식으로 폴백
                            # output과 target이 튜플인 경우 첫 번째 요소만 사용
                            if isinstance(output, tuple):
                                model_output = output[0]
                            else:
                                model_output = output
                                
                            target = fp_inps[index:index+args.batch_size]
                            if isinstance(target, tuple):
                                target = target[0]
                                
                            loss = nn.MSELoss()(model_output, target)
                            
                            if args.aug_loss and fp_inps_2 is not None:
                                aug_target = fp_inps_2[index:index+args.batch_size]
                                if isinstance(aug_target, tuple):
                                    aug_target = aug_target[0]
                                loss += nn.MSELoss()(model_output, aug_target)
                    
                    if not math.isfinite(loss.item()):
                        logger.info("손실이 NAN입니다. 학습을 중단합니다.")
                        break
                        
                    loss_list.append(loss.detach().cpu())
                    optimizer.zero_grad()
                    norm = loss_scaler(loss, optimizer, parameters=get_omni_parameters(qlayer, use_shift)).cpu()
                    norm_list.append(norm.data)
                
                loss_mean = torch.stack(loss_list).mean()
                norm_mean = torch.stack(norm_list).mean()
                logger.info(f"레이어 {i} 반복 {epochs} 손실:{loss_mean} 노름:{norm_mean} 최대 메모리 사용량 {torch.cuda.max_memory_allocated(lm._device) / 1024**2}MB")
            
            clear_temp_variable(qlayer)
            del optimizer
        
        qlayer.half()
        
        # 실제 스무딩 및 양자화
        smooth_and_quant_inplace(qlayer, args, is_llama)
        
        if args.epochs > 0:
            # 양자화 모델의 입력 업데이트
            with torch.no_grad():
                with torch.cuda.amp.autocast() if not args.deactive_amp else nullcontext():
                    for j in range(args.nsamples):
                        outputs = qlayer(quant_inps[j].unsqueeze(0), 
                                      attention_mask=attention_mask,
                                      position_ids=position_ids)
                        # 출력이 튜플인 경우 첫 번째 요소를 찾을 때까지 반복 추출
                        output_tensor = outputs
                        while isinstance(output_tensor, tuple):
                            output_tensor = output_tensor[0]
                        quant_inps[j] = output_tensor
            
            register_scales_and_zeros(qlayer)
            layers[i] = qlayer.to("cpu")
            omni_parameters[i] = omni_state_dict(qlayer)
            torch.save(omni_parameters, os.path.join(args.output_dir, f"omni_parameters.pth"))
        else:
            register_scales_and_zeros(qlayer)
            layers[i] = qlayer.to("cpu")
        
        # 메모리 정리
        del layer
        gc.collect()
        torch.cuda.empty_cache()
    
    logger.info("Matryoshka Quantization 완료!")
    
    return lm 