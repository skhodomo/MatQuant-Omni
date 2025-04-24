CUDA_VISIBLE_DEVICES=0 python main_matquant.py \
--model facebook/opt-125m --eval_ppl \
--epochs 40 --output_dir ./log/opt-125m-w4a16g128 \
--wbits 8 --abits 16 --group_size 128 --lwc --let \
--bit_list "8,4,2" --batch_size 4 --nsamples 128
