FROM nvidia/cuda:10.1-base

RUN apt update && \
    apt install -y bash \
                   build-essential \
                   git \
                   curl \
                   ca-certificates \
	           openjdk-11-jdk-headless \
                   python3 \
                   python3-pip && \
    rm -rf /var/lib/apt/lists

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
    mkl \
    torch \
    spacy \
    pyserini \
    transformers==3.0.2 

WORKDIR /workspace
COPY dpr /workspace/dpr/
COPY *.py /workspace/
COPY index /workspace/index/
COPY reader_checkpoint.cp /workspace/
COPY submission.sh .

CMD ["/bin/bash"] 
