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

