FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# Python 패키지 설치
RUN pip install --no-cache-dir \
    datasets>=2.0.0 \
    einops \
    jsonlines \
    numexpr \
    openai>=0.6.4 \
    omegaconf>=2.2 \
    peft>=0.2.0 \
    pybind11>=2.6.2 \
    pycountry \
    pytablewriter \
    rouge-score>=0.0.4 \
    sacrebleu==1.5.0 \
    scikit-learn>=0.24.1 \
    sqlitedict \
    tqdm-multiprocess \
    zstandard \
    accelerate \
    sentencepiece \
    tokenizers>=0.12.1 \
    torch>=2.0.0 \
    torchvision \
    transformers>=4.31.0 \
    texttable \
    toml \
    attributedict \
    protobuf

# 소스 코드 복사
COPY . /app/

# 기본 명령어 설정
CMD [ "/bin/bash"]