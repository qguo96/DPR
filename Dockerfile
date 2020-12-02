FROM nvidia/cuda:10.1-base

RUN apt update && \
    apt install -y bash \
                   build-essential \
                   curl \
                   ca-certificates \
	           openjdk-11-jdk-headless \
                   python3 \
                   python3-pip && \
    rm -rf /var/lib/apt/lists

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
    pyserini

WORKDIR /workspace
CMD ["/bin/bash"] 
