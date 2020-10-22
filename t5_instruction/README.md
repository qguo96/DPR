This walk-through uses the T5 baseline. We are using the smallest t5.1.1.small_ssm_nq model, for efficiency, but you could replace this with t5.1.1.xl_ssm_nq to get the 3B parameter model.

First, download the EfficientQA development set input examples. We will be using these to test our submission locally. 
This file is not the same as the standard development set file because it does not contain answers. 
If your submission expects an answer field in the input examples it will crash. Please make sure you test your submission with these examples as input.

```
INPUT_DIR=~/efficientqa_input
mkdir ${INPUT_DIR}
wget https://raw.githubusercontent.com/google-research-datasets/natural-questions/master/nq_open/NQ-open.efficientqa.dev.no-annotations.jsonl -P ${INPUT_DIR}
```


Create a submission directory and follow the instructions to download and export a T5 model.


```
# Create the submission directory.
SUBMISSION_DIR=~/t5_efficientqa_submission
MODEL_DIR="${SUBMISSION_DIR}/models"
SRC_DIR="${SUBMISSION_DIR}/src"
mkdir -p "${MODEL_DIR}"
mkdir -p "${SRC_DIR}"

# Install t5
pip install t5

# Select one of the models below by un-commenting it.
MODEL=t5.1.1.small_ssm_nq

git clone https://github.com/google-research/google-research.git
cd google-research/t5_closed_book_qa

# Export the model.
python -m t5.models.mesh_transformer_main \
  --module_import="t5_cbqa.tasks" \
  --model_dir="gs://t5-data/pretrained_models/cbqa/${MODEL}" \
  --use_model_api \
  --mode="export_predict" \
  --export_dir="${MODEL_DIR}/${MODEL}"
```

Our example makes use of tensorflow serving to serve our model. So all we need to do is to create an inference script that will call the model server for each input example, and output predictions in the required format. Create a file predict.py in your ${SRC_DIR} that contains the code below.

```
# Prediction script for T5 running with TF Serving.
from absl import app
from absl import flags

import json
import requests

flags.DEFINE_string('server_host', 'http://localhost:8501', '')
flags.DEFINE_string('model_path', '/v1/models/t5.1.1.small_ssm_nq',
  'Path to model, TF-serving adds the `v1` prefix.')
flags.DEFINE_string('input_path', '', 'Path to input examples.')
flags.DEFINE_string('output_path', '', 'Where to output predictions.')
flags.DEFINE_bool('verbose', True, 'Whether to log all predictions.')

FLAGS = flags.FLAGS


def main(_):
  server_url = FLAGS.server_host + FLAGS.model_path + ':predict'
  with open(FLAGS.output_path, 'w') as fout:
    with open(FLAGS.input_path) as fin:
      for l in fin:
        example = json.loads(l)
        predict_request = '{{"inputs": ["nq question: {0}?"]}}'.format(
            example['question']).encode('utf-8')
        response = requests.post(server_url, data=predict_request)
        response.raise_for_status()
        predicted_answer = response.json()['outputs']['outputs'][0]

        if FLAGS.verbose:
          print('{0} -> {1}'.format(example['question'], predicted_answer))

        fout.write(
            json.dumps(
                dict(question=example['question'], prediction=predicted_answer))
            + '\n')


if __name__ == '__main__':
  app.run(main)
```

We can test this locally using the tensorflow-serving Docker image.

```
docker pull tensorflow/serving:nightly
docker run -t --rm -p 8501:8501 \
  -v ${MODEL_DIR}:/models -e MODEL_NAME=${MODEL} tensorflow/serving:nightly &

python3 "${SRC_DIR}/predict.py" \
  --input_path="${INPUT_DIR}/NQ-open.efficientqa.dev.no-annotations.jsonl" \
  --output_path="/tmp/predictions.jsonl"
```

Now we just need to create our executable submission.sh that will call predict.py. This uses the tensorflow-serving binary, which will be packaged in our Docker submission (instructions below). As with predict.py, create this in your ${SRC_DIR}.

```
# Path to T5 saved model.
MODEL_BASE_PATH='/models'
MODEL_NAME='t5.1.1.small_ssm_nq'
MODEL_PATH="${MODEL_BASE_PATH}/${MODEL_NAME}"

# Get predictions for all questions in the input.
INPUT_PATH=$1
OUTPUT_PATH=$2

# Start the model server and wait, to give it time to come up.
tensorflow_model_server --port=8500 --rest_api_port=8501 \
  --model_name=${MODEL_NAME} --model_base_path=${MODEL_PATH} "$@" &
sleep 20

# Now run predictions on input file.
echo 'Running predictions.'
python predict.py --model_path="/v1/models/${MODEL_NAME}" \
  --verbose=false \
  --input_path=$INPUT_PATH --output_path=$OUTPUT_PATH
echo 'Done predicting.'
```

Make sure that submission.sh is executable, and then create the following dockerfile in ${SUBMISSION_DIR}/Dockerfile. This defines a Docker image that contains all of our code, libraries, and data.

```
ARG TF_SERVING_VERSION=2.3.0
ARG TF_SERVING_BUILD_IMAGE=tensorflow/serving:${TF_SERVING_VERSION}-devel

FROM ${TF_SERVING_BUILD_IMAGE} as build_image
FROM python:3-slim-buster

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install TF Serving pkg.
COPY --from=build_image /usr/local/bin/tensorflow_model_server /usr/bin/tensorflow_model_server

# Install python packages.
RUN pip install absl-py
RUN pip install requests

ADD src .

# The tensorflow serving Docker image expects a model directory at `/models` and
# this will be mounted at `/v1/models`.
ADD models models/
```

Build and then test this Docker image.
```
docker build --tag "$MODEL" "${SUBMISSION_DIR}/."
docker run -v "${INPUT_DIR}:/input" -v "/tmp:/output" "${MODEL}" bash \
  "submission.sh" \
  "input/NQ-open.efficientqa.dev.no-annotations.jsonl" \
  "output/predictions.jsonl"
```
