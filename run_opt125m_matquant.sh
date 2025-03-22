#!/bin/bash

# Docker 컨테이너 안에서 실행할 스크립트
# 출력 디렉토리 생성
mkdir -p matquant_output/opt-125m-matquant

# OPT-125M 모델에 MatQuant 적용
echo "OPT-125M 모델에 MatQuant 적용 중..."
python main_matquant.py \
  --model facebook/opt-125m \
  --abits 16 \
  --group_size 128 \
  --bit_list 8 4 2 \
  --lambda_r 0 1.0 0 \
  --let \
  --lwc \
  --mode quant \
  --epochs 1 \
  --nsamples 128 \
  --batch_size 4 \
  --output_dir ./matquant_output/opt-125m-matquant 