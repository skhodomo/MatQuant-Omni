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
from quantize.matryoshka_omniquant import omniquant_matquant
from quantize.matryoshka_quantizer import Matryoshka_QuantLinear
from categories import subcategories, categories
torch.backends.cudnn.benchmark = True

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

def remove_quantizer_params(model):
    """
    Matryoshka_QuantLinear 모듈에 등록된 양자화 부가 파라미터들을 철저하게 삭제.
    반환: 삭제한 항목 수
    """
    remove_count = 0
    for name, module in model.named_modules():
        if isinstance(module, Matryoshka_QuantLinear):
            # weight_quantizer 내부 삭제
            if hasattr(module, "weight_quantizer"):
                wq = module.weight_quantizer
                for key in ["lowbound_factor", "upbound_factor"]:
                    if key in wq.__dict__:
                        del wq.__dict__[key]
                        remove_count += 1
            # 모듈 자체에 등록된 파라미터나 버퍼, 또는 attribute 삭제
            for attr in [
                "qkv_smooth_scale", "qkv_smooth_shift", "qkt_smooth_scale",
                "out_smooth_scale", "out_smooth_shift",
                "fc1_smooth_scale", "fc1_smooth_shift"
            ]:
                if attr in module._parameters:
                    del module._parameters[attr]
                    remove_count += 1
                if attr in module._buffers:
                    del module._buffers[attr]
                    remove_count += 1
                if attr in module.__dict__:
                    del module.__dict__[attr]
                    remove_count += 1
    return remove_count

@torch.no_grad()
def evaluate(lm, args, logger):
    """
    MatQuant 모델 평가 함수
    """
    results = {}
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
    else:
        if "opt" in args.net.lower():
            lm.model.model.decoder = lm.model.model.decoder.to(lm.device)
        elif "llama" in args.net.lower() or "mixtral" in args.net.lower():
            lm.model = lm.model.to(lm.device)
        elif "falcon" in args.net.lower():
            lm.model.transformer = lm.model.transformer.to(lm.device)

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

            nsamples = testenc.numel() // lm.seqlen
            use_cache = lm.model.config.use_cache
            lm.model.config.use_cache = False
            lm.model.eval()
            nlls = []
            
            for i in tqdm(range(nsamples)):
                batch = testenc[:, (i * lm.seqlen):((i + 1) * lm.seqlen)].to(lm.device)
                if "opt" in args.net.lower():
                    outputs = lm.model.model.decoder(batch)
                elif "llama" in args.net.lower() or "mixtral" in args.net.lower():
                    outputs = lm.model.model(batch)
                elif "falcon" in args.model:
                    outputs = lm.model.transformer(batch)
                
                hidden_states = outputs[0]
                logits = lm.model.lm_head(hidden_states)
                shift_logits = logits[:, :-1, :]
                shift_labels = testenc[:, (i * lm.seqlen):((i + 1) * lm.seqlen)][:, 1:].to(lm.model.lm_head.weight.device)
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                neg_log_likelihood = loss.float() * lm.seqlen
                nlls.append(neg_log_likelihood)
                if i == args.limit:
                    break

            ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * lm.seqlen))
            logger.info(f'{dataset} : {ppl.item()}')
            lm.model.config.use_cache = use_cache
            results[dataset] = ppl.item()

    if args.tasks != "":
        t_results = evaluator.simple_evaluate(
            lm,
            tasks=args.tasks,
            num_fewshot=args.num_fewshot,
            limit=None if args.limit == -1 else args.limit,
        )
        results.update(t_results)
        logger.info(results)
        pprint(results)
        
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
                logger.info("평균 정확도 {:.4f} - {}".format(cat_acc, cat))
            weighted_acc = np.mean(all_cors)
            logger.info("전체 평균 정확도: {:.4f}".format(weighted_acc))

    return results

def calculate_perplexity(model, testenc, dev):
    """
    퍼플렉시티 계산
    """
    nlls = []
    try:
        num_rows = testenc.num_rows
    except AttributeError:
        num_rows = len(testenc)
    for i in range(num_rows):
        batch = testenc[i:i+1]
        if isinstance(batch, dict) and "input_ids" in batch:
            batch = batch["input_ids"]
        if not torch.is_tensor(batch):
            try:
                batch = torch.tensor(batch)
            except Exception as e:
                raise RuntimeError(f"배치를 tensor로 변환할 수 없습니다. batch type: {type(batch)}") from e
        batch = batch.to(dev)
        with torch.no_grad():
            outputs = model(batch)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            shift_logits = outputs[..., :-1, :].contiguous()
            shift_labels = batch[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            neg_log_likelihood = loss.float() * batch.shape[1]
            nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / num_rows)
    return ppl.item()

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
    parser.add_argument("--let_lr", default=5e-3, type=float, help="LET 학습률")
    parser.add_argument("--lwc_lr", default=1e-2, type=float, help="LWC 학습률")
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
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
    
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
        
        # === 양자화 부가 파라미터 제거 ===
        remove_count = remove_quantizer_params(lm.model)
        logger.info(f"총 {remove_count}개의 부가 파라미터 삭제 완료.")
        logger.info(f"모델을 {args.save_dir}에 저장 중...")
        lm.model.save_pretrained(args.save_dir)
        lm.tokenizer.save_pretrained(args.save_dir)
        del lm
        torch.cuda.empty_cache()
        logger.info("기존 모델 해제 및 GPU 메모리 정리 완료")
        logger.info("저장된 모델로 재로딩 시작...")
        lm = LMClass.from_pretrained(args.save_dir, args=args)
        lm.seqlen = 2048
        lm.model.eval()
        for param in lm.model.parameters():
            param.requires_grad = False
        if torch.cuda.is_available():
            lm._device = "cuda:0"
        else:
            lm._device = "cpu"
        logger.info(f"재로딩된 모델이 {lm._device}에서 실행됩니다.")
    
    elif args.mode == "eval":
        logger.info("저장된 모델로 재로딩 시작...")
        lm = LMClass.from_pretrained(args.save_dir, args=args)
        lm.seqlen = 2048
        lm.model.eval()
        for param in lm.model.parameters():
            param.requires_grad = False
        if torch.cuda.is_available():
            lm._device = "cuda:0"
        else:
            lm._device = "cpu"
        logger.info(f"재로딩된 모델이 {lm._device}에서 실행됩니다.")
        cache_testenc = f'{args.cache_dir}/testenc_{args.model_family}_{args.calib_dataset}_{args.nsamples}.cache'
        if os.path.exists(cache_testenc):
            testenc = torch.load(cache_testenc)
            logger.info(f"캐시에서 테스트 데이터 로드: {cache_testenc}")
        else:
            _, testenc = get_loaders(args.calib_dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=lm.seqlen)
            torch.save(testenc, cache_testenc)
    
    else:
        logger.error("잘못된 모드입니다. --mode는 quant 또는 eval 이어야 합니다.")
        sys.exit(1)
    
    results = evaluate(lm, args, logger)
    
    if args.save_dir:
        logger.info(f"모델을 {args.save_dir}에 최종 저장 중...")
        lm.model.save_pretrained(args.save_dir)
        lm.tokenizer.save_pretrained(args.save_dir)
        with open(os.path.join(args.save_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(results, f, indent=4)
    
    with open(os.path.join(args.output_dir, 'matquant_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    logger.info(f"최종 모델이 {args.output_dir}에 저장되었습니다.")
    logger.info("Matryoshka Quantization 완료!")

if __name__ == '__main__':
    main()