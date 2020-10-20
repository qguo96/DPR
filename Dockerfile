FROM huggingface/transformers-pytorch-gpu:3.0.2
RUN apt-get update
RUN apt-get install -y openjdk-11-jdk-headless
RUN pip3 install spacy pyserini
COPY dpr /workspace/dpr/
COPY *.py /workspace/
COPY index /workspace/index/
COPY reader_checkpoint.cp /workspace/
COPY submission.sh .
