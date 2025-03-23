#!/bin/bash

# Docker 이미지 빌드
echo "MatQuant Docker 이미지 빌드 중..."
docker build -t matquant:latest .

# Docker 컨테이너 실행
echo "MatQuant Docker 컨테이너 실행 중..."
docker run --gpus all -it \
  --name matquant_container \
  -v $(pwd):/app \
  matquant:latest