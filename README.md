# Seq2Seq Corefernce Resolution

This code is based on:
1. https://github.com/yuvalkirstain/s2e-coref
2. https://github.com/huggingface/transformers/tree/master/examples/pytorch/summarization

## Main Code Parts

From indices to sequence:
https://github.com/eladb3/seq2seq_coref/blob/7ee2908a58c245b6a051bb75d50c4678bf73ff08/DataAugmentation/data_aug.py#L257

From sequence to indices:
https://github.com/eladb3/seq2seq_coref/blob/7ee2908a58c245b6a051bb75d50c4678bf73ff08/DataAugmentation/data_aug.py#L287
## Run

Install env from `requirements.txt`

Prepare the data (follow https://github.com/yuvalkirstain/s2e-coref), your data dir should contain the files train.english.jsonlines, dev.english.jsonlines and test.english.jsonlines.

Next run:
```
python run_summarization.py
--model_name_or_path t5-base \
--data_root $PATH_TO_DATA_ROOT \
--do_eval --do_predict --do_train \
--train_file dsasdass.json \
--test_file sdad.json \
--validation_file sada.json \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 2 \
--log_level info \
--max_source_length 768 \
--max_target_length 768 \
--max_sentences 7 \
--add_speakers True \
--augmentor naive_augmentor \
--start_end_tokens_augmentor_max_clusters 25 \
--num_train_epochs 15 \
--output_dir ./myexp \
--predict_with_generate \
--naive_augmentor_add_loc_tokens True \
--replace_loc_tokens True
```
