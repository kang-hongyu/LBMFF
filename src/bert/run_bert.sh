#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES='1'
python run_classifier.py \
  --task_name="mrpc"  \
  --do_lower_case=True \
  --crf=False \
  --do_train=False \
  --do_eval=False   \
  --do_predict=True \
  --data_dir=../data/TL-HGBI/data \
  --filename=BLA103792.txt \
  --vocab_file=../bert/multilingual_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=../bert/multilingual_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=../bert/model_pretrain_drug_disease/model.ckpt-800000 \
  --max_seq_length=128   \
  --train_batch_size=32   \
  --learning_rate=2e-5   \
  --num_train_epochs=10   \
  --output_dir=../data/model_bert_TL-HGBI1 \

