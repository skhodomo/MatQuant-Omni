python main_matquant.py \
--model facebook/opt-125m  \
--epochs 0 --output_dir ./log/test \
--eval_ppl --wbits 4 --abits 16 --group_size 128 --lwc \
--resume ./matquant_output/opt-125m-matquant/checkpoint-final.pt \
--mode eval \
--nsamples 1 \
--batch_size 1 \
