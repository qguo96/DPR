#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import sys
import argparse
import csv
import json
import numpy as np
import torch

from dpr.options import add_encoder_params, setup_args_gpu, set_encoder_params_from_state, \
            add_tokenizer_params, add_training_params, add_reader_preprocessing_params

if __name__ == '__main__':
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()

    add_encoder_params(parser)
    add_training_params(parser)
    add_tokenizer_params(parser)
    add_reader_preprocessing_params(parser)


    parser.add_argument("--max_n_answers", default=10, type=int,
                        help="Max amount of answer spans to marginalize per singe passage")
    parser.add_argument('--passages_per_question', type=int, default=2,
                        help="Total amount of positive and negative passages per question")
    parser.add_argument('--passages_per_question_predict', type=int, default=50,
                        help="Total amount of positive and negative passages per question for evaluation")
    parser.add_argument("--max_answer_length", default=10, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument('--eval_top_docs', nargs='+', type=int,
                        help="top retrival passages thresholds to analyze prediction results for")
    parser.add_argument('--checkpoint_file_name', type=str, default='dpr_reader')
    parser.add_argument('--prediction_results_file', type=str, help='path to a file to write prediction results to')

    args = parser.parse_args()
    setup_args_gpu(args)
    print(args.n_gpu)
    args.dev_file = 'retrieval_result.json'
    args.dev_batch_size = 32
    args.pretrained_model_cfg = 'bert-base-uncased'
    args.encoder_model_type = 'hf_bert'
    args.sequence_length = 350
    args.do_lower_case = True
    args.eval_top_docs = 50
    args.passages_per_question_predict = 50
    
    #from IPython import embed; embed()
    from train_reader import ReaderTrainer

    class MyReaderTrainer(ReaderTrainer):
        def _save_predictions(self, out_file, prediction_results):
            with open(out_file, 'w', encoding="utf-8") as output:
                save_results = []
                for r in prediction_results:
                    save_results.append({
                        'question': r.id,
                        'prediction': r.predictions[args.passages_per_question_predict].prediction_text
                    })
                output.write(json.dumps(save_results, indent=4) + "\n")
    
    trainer = MyReaderTrainer(args)
    trainer.reader = torch.load('reader_checkpoint.cp')
    trainer.reader.cuda()
    
    trainer.validate()

    #os.remove(retrieval_file)
    #for i in range(args.num_workers):
    #    os.remove(retrieval_file.replace(".json", ".{}.pkl".format(i)))

