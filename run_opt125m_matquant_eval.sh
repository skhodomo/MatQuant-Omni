python main_matquant.py \
--model facebook/opt-125m  \
--epochs 0 --output_dir ./log/test \
--eval_ppl --wbits 4 --abits 16 --group_size 128 --lwc \
--mode eval \
--nsamples 128 \
--batch_size 1 \
--bit_list 8 4 2 \
--save_dir ./quantized_models/opt-125m-matquant
