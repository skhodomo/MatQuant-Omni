import os
import sys
import random
import numpy as np
from models.LMClass import LMClass
import torch
import time
from datautils import get_loaders
from lm_eval import evaluator
from pprint import pprint
from parallel_utils import map_layers_to_multi_gpus, get_lowest_occupied_gpu
import torch.nn as nn
from quantize.matryoshka_omniquant import omniquant
from tqdm import tqdm
import utils
from pathlib import Path
from categories import subcategories, categories
from quantize.utils import change_n_bits
from models.int_llama_layer import QuantLlamaDecoderLayer
from models.int_opt_layer import QuantOPTDecoderLayer
from quantize.int_linear import QuantLinear
import copy
import pdb
import math

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

@torch.no_grad()
def evaluate(lm, args, logger):

        
    def change_bit_all_layers(lm, bit):
        """
        lm 모델 내 모든 QuantLinear에 대해 비트 슬라이싱 quantization 적용.
        group quantization, padding(deficiency), slicing 모두 지원.
        """
        if "llama" in lm.model.config.model_type.lower() or "mixtral" in lm.model.config.model_type.lower():
            layers = lm.model.model.layers
        elif "opt" in lm.model.config.model_type.lower():
            layers = lm.model.model.decoder.layers
        elif "falcon" in lm.model.config.model_type.lower():
            layers = lm.model.transformer.h
        else:
            raise NotImplementedError("Only llama/opt/falcon/mixtral supported.")

        def bit_slice(qc, r, c=8):
            shift = 2 ** (c - r)
            sliced = torch.round(qc / shift)
            sliced = torch.clamp(sliced, 0, 2 ** r - 1)
            return sliced * shift

        for i, layer in enumerate(layers):
            for name, module in layer.named_modules():
                if isinstance(module, QuantLinear):
                    quantizer = module.weight_quantizer
                    weight_fp = module.weight
                    dim0, dim1 = weight_fp.shape
                    group_size = getattr(quantizer, "group_size", None)
                    pad = getattr(quantizer, "deficiency", 0)

                    # scale / zp (round_zero_point)
                    scale = getattr(quantizer, "scale", getattr(quantizer, "scales", None))
                    zp = getattr(quantizer, "round_zero_point", getattr(quantizer, "zeros", None))

                    if scale is None:
                        print(f"⚠️ [L{i}] {name} - scale not found, skipping")
                        continue
                    if zp is None and not quantizer.disable_zero_point:
                        print(f"⚠️ [L{i}] {name} - zero point not found, skipping")
                        continue

                    try:
                        if group_size is not None:
                            # Group-wise quantization
                            padded_dim1 = dim1 + pad
                            num_groups = math.ceil(padded_dim1 / group_size)

                            scale = scale.view(dim0, num_groups, 1)  # [out, groups, 1]
                            zp = zp.view(dim0, num_groups, 1) if zp is not None else None

                            if pad > 0:
                                weight_fp = torch.cat(
                                    [weight_fp, torch.zeros((dim0, pad), dtype=weight_fp.dtype, device=weight_fp.device)],
                                    dim=1
                                )

                            weight_grouped = weight_fp.view(dim0, num_groups, group_size)

                            # dequantization
                            dequant = (weight_grouped / scale) + (zp if zp is not None else 0)
                            weight_int = dequant.round().clamp(quantizer.qmin, quantizer.qmax)

                            # slicing
                            if bit == 8:
                                sliced = weight_int
                            elif bit == 4:
                                sliced = bit_slice(weight_int, 4)
                            elif bit == 2:
                                sliced = bit_slice(bit_slice(weight_int, 4), 2)
                            else:
                                raise ValueError(f"Unsupported bit-width: {bit}")

                            requant = (sliced - zp) * scale if zp is not None else sliced * scale
                            requant = requant.view(dim0, padded_dim1)
                            if pad > 0:
                                requant = requant[:, :-pad]

                        else:
                            # Per-channel quantization
                            dequant = (weight_fp / scale) + (zp if zp is not None else 0)
                            weight_int = dequant.round().clamp(quantizer.qmin, quantizer.qmax)

                            if bit == 8:
                                sliced = weight_int
                            elif bit == 4:
                                sliced = bit_slice(weight_int, 4)
                            elif bit == 2:
                                sliced = bit_slice(bit_slice(weight_int, 4), 2)
                            else:
                                raise ValueError(f"Unsupported bit-width: {bit}")

                            requant = (sliced - zp) * scale if zp is not None else sliced * scale

                        module.weight = requant.to(weight_fp.dtype)
                        print(f"[✓] Layer {i} - {name}: bit {bit}, dtype: {module.weight.dtype}")

                    except Exception as e:
                        print(f"❌ Failed at Layer {i} - {name}: {e}")
                        continue
            
        
    print("평가 시작")
    #weight , scale , round_zero_point 출력
    for name, module in lm.model.model.named_modules():
        if isinstance(module, QuantLinear):
            print(f"\n--- {name} ---")
            print(f"weight dtype: {module.weight.dtype}, shape: {module.weight.shape}")

            if hasattr(module.weight_quantizer, "scale"):
                print(f"scale (learnable or temp): exists → shape: {module.weight_quantizer.scale.shape}")
            elif hasattr(module.weight_quantizer, "scales"):
                print(f"scale (buffer): exists → shape: {module.weight_quantizer.scales.shape}")
            else:
                print("scale: ❌ not found")

            if hasattr(module.weight_quantizer, "round_zero_point"):
                print(f"zero_point (learnable or temp): exists → shape: {module.weight_quantizer.round_zero_point.shape}")
            elif hasattr(module.weight_quantizer, "zeros"):
                print(f"zero_point (buffer): exists → shape: {module.weight_quantizer.zeros.shape}")
            else:
                print("zero_point: ❌ not found") 
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

    results = {}
    # 평가할 데이터셋 (예: wikitext2, c4)
    for dataset in ["wikitext2", "c4"]:
        cache_testloader = f'{args.cache_dir}/testloader_{args.model_family}_{dataset}_all.cache'
        if os.path.exists(cache_testloader):
            testloader = torch.load(cache_testloader)
            logger.info(f"load calibration from {cache_testloader}")
        else:
            dataloader, testloader = get_loaders(
                dataset,
                seed=args.seed,
                model=args.model,
                seqlen=lm.seqlen,
            )
            torch.save(testloader, cache_testloader)
        # testloader 구조에 따라 testenc 설정
        if "c4" in dataset:
            testenc = testloader
        else:
            testenc = testloader.input_ids

        nsamples = testenc.numel() // lm.seqlen
        use_cache = lm.model.config.use_cache
        lm.model.config.use_cache = False
        lm.model.eval()

        # 원래 모델 상태 백업
        original_state = copy.deepcopy(lm.model.state_dict())

        # 각 비트에 대해 평가
        for bit in args.bit_list:
            logger.info(f"Evaluating {dataset} with {bit}-bit quantization...")
            
            # 비트별 파라미터 파일 로드
            bit_params_path = os.path.join(args.output_dir, f"omni_parameters_{bit}bit.pth")
            if os.path.exists(bit_params_path):
                bit_params = torch.load(bit_params_path)
                
                # 모델의 각 레이어에 파라미터 적용
                if "llama" in args.net.lower() or "mixtral" in args.net.lower():
                    layers = lm.model.model.layers
                elif "opt" in args.net.lower():
                    layers = lm.model.model.decoder.layers
                elif "falcon" in args.net.lower():
                    layers = lm.model.transformer.h
                for i, layer in enumerate(layers):
                    if i in bit_params:
                        layer.load_state_dict(bit_params[i], strict=False)

                # 🔥 여기서 Matryoshka-style 비트 슬라이싱 적용!
                if bit != 8:
                    change_bit_all_layers(lm, bit)
                
             
            
            nlls = []
            for i in range(nsamples):
                batch = testenc[:, (i * lm.seqlen):((i + 1) * lm.seqlen)].to(lm.device)
                # 모델 종류에 따라 호출
                if "opt" in args.net.lower():
                    outputs = lm.model.model.decoder(batch)
                elif "llama" in args.net.lower() or "mixtral" in args.net.lower():
                    outputs = lm.model.model(batch)
                elif "falcon" in args.model.lower():
                    outputs = lm.model.transformer(batch)
                else:
                    raise ValueError("Unsupported model for evaluation.")

                # outputs[0]가 hidden_states라 가정합니다.
                hidden_states = outputs[0]
                logits = lm.model.lm_head(hidden_states)
                shift_logits = logits[:, :-1, :]
                shift_labels = testenc[:, (i * lm.seqlen):((i + 1) * lm.seqlen)][:, 1:].to(lm.model.lm_head.weight.device)
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                neg_log_likelihood = loss.float() * lm.seqlen
                nlls.append(neg_log_likelihood)
                if i == args.limit:
                    break

            # Perplexity 계산: 전체 NLL을 nsamples * seqlen으로 나눈 뒤 exp를 취함
            ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * lm.seqlen))
            logger.info(f'{dataset} | {bit}-bit: {ppl.item()}')
            results[f"{dataset}_{bit}bit"] = ppl.item()

            # 모델 상태 복원
            lm.model.load_state_dict(original_state)

        results[dataset] = results.get(f"{dataset}_{args.bit_list[0]}bit")
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
        # for test of MMLU
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
                logger.info("Average accuracy {:.4f} - {}".format(cat_acc, cat))
            weighted_acc = np.mean(all_cors)
            logger.info("Average accuracy: {:.4f}".format(weighted_acc))               
    return results
def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model name of model path")
    parser.add_argument("--cache_dir", default="./cache", type=str, help="cache dir of dataset, leading to faster debug")
    parser.add_argument("--output_dir", default="../log/", type=str, help="direction of logging file")
    parser.add_argument("--save_dir", default=None, type=str, help="direction for saving fake quantization model")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--real_quant", default=False, action="store_true", help="real quantization, which can see memory reduce. Note that due to the limitations of AutoGPTQ kernels, the real quantization of weight-only quantization can only lead memory reduction, but with slower inference speed.")
    parser.add_argument("--calib_dataset",type=str,default="wikitext2",
        choices=["wikitext2", "ptb", "c4", "mix","pile"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument("--nsamples", type=int, default=128, help="Number of calibration data samples.")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size.")
    parser.add_argument("--seed", type=int, default=2, help="Seed for sampling the calibration data.")
    parser.add_argument("--tasks", default="")
    parser.add_argument("--eval_ppl", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--wbits", type=int, default=4)
    parser.add_argument("--abits", type=int, default=16)
    parser.add_argument("--group_size", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--let_lr", type=float, default=5e-3)
    parser.add_argument("--lwc_lr", type=float, default=1e-2)
    parser.add_argument("--wd", type=float, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--let",default=False, action="store_true",help="activate learnable equivalent transformation")
    parser.add_argument("--lwc",default=False, action="store_true",help="activate learnable weight clipping")
    parser.add_argument("--aug_loss", default=False, action="store_true", help="calculate additional loss with same input")
    parser.add_argument("--symmetric",default=False, action="store_true", help="symmetric quantization")
    parser.add_argument("--disable_zero_point",default=False, action="store_true", help="quantization without zero_point")
    parser.add_argument("--a_dynamic_method", type=str, default="per_token", choices=["per_token"])
    parser.add_argument("--w_dynamic_method", type=str, default="per_channel", choices=["per_channel"])
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--multigpu", action="store_true", help="at eval, map model to multiple gpus")
    parser.add_argument("--deactive_amp", action="store_true", help="deactivate AMP when 8<=bits<16")
    parser.add_argument(
        "--attn_implementation",
        type=str, required=False, default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="attention implementation that the model works with",
    )
    parser.add_argument("--net", type=str, default=None, choices=net_choices)
    parser.add_argument("--act-scales", type=str, default=None)
    parser.add_argument("--act-shifts", type=str, default=None)
    parser.add_argument("--bit_list", type=lambda x: [int(i) for i in x.split(',')], default=[8,4,2], 
                        help="Comma-separated list of bits for matryoshka quantization")

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # check
    if args.epochs > 0:
        assert args.lwc or args.let
        
    if (args.wbits<16 and args.wbits>=8) or (args.abits<16 and args.abits>=8):
        args.deactive_amp = True

    # init logger
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.cache_dir:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    logger = utils.create_logger(output_dir)
    logger.info(args)
    
    # load model
    if args.net is None:
        args.net = args.model.split('/')[-1]
    # assert args.net in net_choices
    args.model_family = args.net.split('-')[0]
    lm = LMClass(args)
    lm.seqlen = 2048
    lm.model.eval()
    for param in lm.model.parameters():
        param.requires_grad = False

    

    args.weight_quant_params = {
        "n_bits": args.bit_list[0],  # 기본 비트 수를 bit_list의 첫 번째 값으로 설정
        "per_channel_axes": [0],
        "symmetric": args.symmetric,
        "dynamic_method": args.w_dynamic_method,
        "group_size": args.group_size,
        "lwc":args.lwc,
        "disable_zero_point": args.disable_zero_point
    }
    args.act_quant_params = {
        "n_bits":  args.abits,
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

    if args.multigpu:
        gpu_id = get_lowest_occupied_gpu(wait_memory=5000)
        lm._device = f"cuda:{gpu_id}"
        logger.info(f"set quantization in gpu {gpu_id}")

    # act scales and shifts
    if args.act_scales is None:
        args.act_scales = f'./act_scales/{args.net}.pt'
    if args.act_shifts is None:
        args.act_shifts = f'./act_shifts/{args.net}.pt'

    # quantization
    if args.wbits < 16 or args.abits <16:
        logger.info("=== start quantization ===")
        tick = time.time()     
        # load calibration dataset
        cache_dataloader = f'{args.cache_dir}/dataloader_{args.model_family}_{args.calib_dataset}_{args.nsamples}.cache'
        if os.path.exists(cache_dataloader):
            dataloader = torch.load(cache_dataloader)
            logger.info(f"load calibration from {cache_dataloader}")
        else:
            dataloader, _ = get_loaders(
                args.calib_dataset,
                nsamples=args.nsamples,
                seed=args.seed,
                model=args.model,
                seqlen=lm.seqlen,
            )
            torch.save(dataloader, cache_dataloader)    
        act_scales = None
        act_shifts = None
        if args.let:
            act_scales = torch.load(args.act_scales)
            act_shifts = torch.load(args.act_shifts)
        omniquant(
            lm,
            args,
            dataloader,
            act_scales,
            act_shifts,
            logger,
        )
        logger.info(time.time() - tick)
    if args.save_dir:
        # delete omni parameters
        for name, module in lm.model.named_modules():
            if isinstance(module, QuantLinear):
                del module.weight_quantizer.lowbound_factor
                del module.weight_quantizer.upbound_factor
            if isinstance(module,QuantLlamaDecoderLayer) or isinstance(module,QuantOPTDecoderLayer) or isinstance(module,QuantOPTDecoderLayer):
                if args.let:
                    del module.qkv_smooth_scale
                    del module.qkv_smooth_shift
                    del module.out_smooth_scale
                    del module.out_smooth_shift
                    del module.fc1_smooth_scale
                    del module.fc1_smooth_shift           
        lm.model.save_pretrained(args.save_dir)  
        lm.tokenizer.save_pretrained(args.save_dir) 
    evaluate(lm, args,logger)


if __name__ == "__main__":
    print(sys.argv)
    main()
