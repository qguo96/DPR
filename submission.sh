INPUT_PATH=$1
OUTPUT_PATH=$2

python3 retrieval.py --qa ${INPUT_PATH}
python3 reader.py --prediction_results_file ${OUTPUT_PATH}