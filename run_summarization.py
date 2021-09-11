#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset, load_metric

import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from mytrainer import Seq2SeqTrainer
from transformers.data.data_collator import DataCollatorForSeq2Seq
from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
import DataAugmentation as DA
from utils import distributed as du
import json
import torch
import pickle
datasets.disable_progress_bar()
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.10.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    dropout: float = field(
        default=-1,
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    text_column: Optional[str] = field(
        default='text',
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    summary_column: Optional[str] = field(
        default='cor',
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    train_file: Optional[str] = field(
        default='dsad.json', metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default='dsad.json',
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (rouge) on "
            "(a jsonlines or csv file)."
        },
    )
    test_file: Optional[str] = field(
        default='dsad.json',
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on " "(a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=0,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=0,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=0,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    augmentor: Optional[str] = field(
        default='naive_augmentor', metadata={"help": "Augmentor for coreference data"}
    )

    start_end_tokens_augmentor_max_clusters: Optional[int] = field(
        default=25, metadata={"help": ""}
    )

    start_end_tokens_augmentor_shuffle_cluster_indices: Optional[bool] = field(
        default=False, metadata={"help": ""}
    )
    start_end_tokens_augmentor_better_alignment: Optional[bool] = field(
        default=False, metadata={"help": ""}
    )
    eval_coref_from_file: Optional[bool] = field(
        default=False, metadata={"help": ""}
    )
    naive_augmentor_add_loc_tokens: Optional[bool] = field(
        default=False, metadata={"help": ""}
    )
    naive_augmentor_threshold: Optional[float] = field(
        default=0.8, metadata={"help": ""}
    )
    empty_debug: Optional[bool] = field(
        default=False, metadata={"help": ""}
    )


    replace_loc_tokens: Optional[bool] = field(
        default=False, metadata={"help": ""}
    )


    max_sentences: Optional[int] = field(
        default=10, metadata={"help": "max sentences in input text"}
    )
    add_speakers: Optional[bool] = field(
        default=False, metadata={"help": "add speakers annotation to input seq"}
    )

    mydebug: Optional[bool] = field(
        default=False, metadata={"help": "debug"}
    )

    hard_debug: Optional[bool] = field(
        default=False, metadata={"help": "debug"}
    )

    from_scratch: Optional[bool] = field(
        default=False, metadata={"help": "debug"}
    )

    load_model_path: Optional[str] = field(
        default='', metadata={"help": "debug"}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
}


def filter_and_log(name, ds, filter, logger):
    n_before = len(ds)
    ds = ds.filter(filter)
    n_after = len(ds)
    logger.info(f"filtered {n_before-n_after} samples from {name}")
    return ds
def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        # import os
        # os.environ['CUDA_VISIBLE_DEVICES'] = '2'
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    DEBUG = data_args.mydebug
    HARD_DEBUG = data_args.hard_debug
    training_args.do_eval = True
    if data_args.from_scratch:
        model_args.model_name_or_path = 'facebook/bart-base'
    # if DEBUG:
    #     training_args.device='cpu'

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    training_args.remove_unused_columns = True

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    logger.info(f"Preparing coreference data...")
    augmentor = getattr(DA, data_args.augmentor)(logger, args = data_args, add_speakers = data_args.add_speakers, max_sentences = data_args.max_sentences, debug=2 if HARD_DEBUG else (1 if DEBUG else 0))
    data_args.train_file, data_args.validation_file, data_args.test_file = augmentor.prepare_all_splits()
    logger.info(f"Training/evaluation parameters {training_args}")

    if data_args.source_prefix is None and model_args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            logger.warn(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
        extension = data_args.train_file.split(".")[-1]
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
        extension = data_args.validation_file.split(".")[-1]
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file
        extension = data_args.test_file.split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    if model_args.dropout >= 0:
        config.dropout = model_args.dropout
    # add special tokens
    tokenizer.add_tokens(list(augmentor.const.values()))
    max_txt ,max_sum = augmentor.get_max_len(tokenizer)
    logger.info(f"max_txt ,max_sum: {max_txt ,max_sum}")

    if data_args.max_source_length <= 0:
        max_model_length = tokenizer.model_max_length
        data_args.max_source_length = max_model_length
        data_args.max_target_length = max_model_length
        data_args.val_max_target_length = max_model_length
    # data_args.val_max_target_length = data_args.max_target_length
    logger.info(f"data_args.max_source_length: {data_args.max_source_length}")
    logger.info(f"data_args.max_target_length: {data_args.max_target_length}")
    logger.info(f"data_args.val_max_target_length: {data_args.val_max_target_length}")


    if data_args.from_scratch:
        from transformers import BartForConditionalGeneration
        model = BartForConditionalGeneration(config=config)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""
    if prefix != "":
        prefix = f'"{prefix}: "'
    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # Get the column names for input/target.
    dataset_columns = summarization_name_mapping.get(data_args.dataset_name, None)
    if data_args.text_column is None:
        text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        text_column = data_args.text_column
        if text_column not in column_names:
            raise ValueError(
                f"--text_column' value '{data_args.text_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.summary_column is None:
        summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        summary_column = data_args.summary_column
        if summary_column not in column_names:
            raise ValueError(
                f"--summary_column' value '{data_args.summary_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    def preprocess_function(examples, idx):
        inputs = examples[text_column]
        targets = examples[summary_column]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding)#, truncation=True)
        # {'input_ids':List[int], 'attention_mask':List[int]}

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length + 2, padding=padding)#, truncation=True)
            # {'input_ids':List[int], 'attention_mask':List[int]}
        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        model_inputs['indices'] = idx

        return model_inputs

    def _filter(sample):
        n_input = len(sample['input_ids'])
        n_labels = len(sample['labels'])
        return n_input <= data_args.max_source_length and n_labels <= data_args.max_target_length + 2

    if not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)

        check=os.path.join(training_args.output_dir, 'pytorch_model.bin')
        pload = None
        if os.path.isfile(check):
            pload = check
        elif last_checkpoint:
            pload = os.path.join(last_checkpoint, 'pytorch_model.bin')
        elif data_args.load_model_path and os.path.isfile(data_args.load_model_path):
            pload = data_args.load_model_path

        if pload:
            logger.info(f"ELAD: Loading model weights from {pload}")
            model.load_state_dict(torch.load(pload, map_location='cpu'))
        else:
            logger.info(f"ELAD: didn't find model to load")

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
                with_indices=True,
            )
        train_dataset = filter_and_log("train_dataset", train_dataset, _filter, logger)

        if data_args.max_train_samples > 0:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
                with_indices=True,
            )
        eval_dataset = filter_and_log("eval_dataset", eval_dataset, _filter, logger)
        if data_args.max_eval_samples > 0:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
                with_indices=True,
            )
        predict_dataset = filter_and_log("predict_dataset", predict_dataset, _filter, logger)
        if data_args.max_predict_samples > 0:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    # Metric
    metric = load_metric("rouge")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        # Extract a few results from ROUGE
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    )

    training_args.num_beams = data_args.num_beams
    training_args.val_max_target_length = data_args.val_max_target_length

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples  > 0 else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # metrics = trainer.evaluate(
        #     max_length=data_args.val_max_target_length, num_beams=data_args.num_beams, metric_key_prefix="eval"
        # )
        # max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples > 0 else len(eval_dataset)
        # metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        # trainer.log_metrics("eval", metrics)
        # trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        # for name, current_ds in [('predict_test', predict_dataset), ('predict_dev', eval_dataset)]:
        for name, current_ds in [('predict_dev', eval_dataset)]:
            logger.info("*** Predict ***")
            output_prediction_file = os.path.join(training_args.output_dir, f"{name}_beams_{data_args.num_beams}_generated_predictions.txt")
            outpath = os.path.join(training_args.output_dir, f"{name}_beams_{data_args.num_beams}_generated_predictions.pkl")
            if not data_args.eval_coref_from_file:
                predict_results = trainer.predict(
                    current_ds,
                    metric_key_prefix=name,
                    max_length=data_args.val_max_target_length,
                    num_beams=data_args.num_beams,
                )
                metrics = predict_results.metrics
                max_predict_samples = (
                    data_args.max_predict_samples if data_args.max_predict_samples > 0 else len(predict_dataset)
                )
                metrics[f"{name}_samples"] = min(max_predict_samples, len(current_ds))

                trainer.log_metrics(f'{name}_beams_{data_args.num_beams}', metrics)
                trainer.save_metrics(f'{name}_beams_{data_args.num_beams}', metrics)

                if trainer.is_world_process_zero():
                    if training_args.predict_with_generate:
                        predictions = tokenizer.batch_decode(
                            predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                        )
                        predictions = [pred.strip() for pred in predictions]

                        
                        with open(output_prediction_file, "w") as writer:
                            writer.write("\n".join(predictions))
                    with open(outpath, 'wb') as f:
                        pickle.dump(predict_results, f)
            else:
                with open(outpath, 'rb') as f:
                    predict_results = pickle.load(f)
            if du.is_root_proc():
                eval_coref(logger, predict_results,current_ds, name, tokenizer, augmentor, training_args, data_args)
    if training_args.push_to_hub:
        kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "summarization"}
        if data_args.dataset_name is not None:
            kwargs["dataset_tags"] = data_args.dataset_name
            if data_args.dataset_config_name is not None:
                kwargs["dataset_args"] = data_args.dataset_config_name
                kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
            else:
                kwargs["dataset"] = data_args.dataset_name

        trainer.push_to_hub(**kwargs)


    return results

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()



##

from myeval import Evaluator as Coref_Evaluator
def eval_coref(logger, predict_results,dataset, split, tokenizer, augmentor, training_args, data_args):
    logger.info(f"*** Evaluating coref for {split} ***")


    logger.info(" >> Generating predictions ")

    metrics = predict_results.metrics
    max_predict_samples = (
        data_args.max_predict_samples if data_args.max_predict_samples > 0 else len(dataset)
    )
    metrics["predict_samples"] = min(max_predict_samples, len(dataset))

    predictions = tokenizer.batch_decode(
        predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    predictions = [pred.strip() for pred in predictions]

    labels = predict_results.label_ids
    labels[labels < 0] = 0
    labels = tokenizer.batch_decode(
        labels, skip_special_tokens=True, clean_up_tokenization_spaces=True,
    )
    labels = [l.strip() for l in labels]

    indices = predict_results.indices

    outdir = os.path.join(training_args.output_dir, f'{split}_coref_eval')
    os.makedirs(outdir, exist_ok=True, mode=0o777)
    output_prediction_file = os.path.join(outdir, f"_{split}_generated_predictions.txt")
    output_prediction_file_unified = os.path.join(outdir, f"_{split}_generated_predictions_unified.txt")

    lines = [{"idx":i, "prediction":p, "label": l} for i, p, l in zip(indices.tolist(), predictions, labels)]

    lines, unified_lines = augmentor.unaugment('dev' if split.endswith('dev') else 'test', lines) # [{doc_key:str, gold_clusters:List[List[List[int]]], predicted_clusters:List[List[List[int]]], prediction:str, label:str}]

    lines = {ll['doc_key']:ll for ll in lines}
    lines = list(lines.values())
    logger.info(f"Evaluatin coref on {len(lines)} samples")
    with open(output_prediction_file, "wt") as f:
        for l in lines:
            f.write(json.dumps(l) + '\n')
    with open(output_prediction_file_unified, "wt") as f:
        for l in unified_lines:
            f.write(json.dumps(l) + '\n')

    logger.info(f" >> Predictions saved to {output_prediction_file} ")
    logger.info(f" >> Predictions saved to {output_prediction_file_unified} ")

    logger.info(" >>  Evaluating with splits")
    coref_evaluator = Coref_Evaluator(logger, outdir, experiment_name=split)
    results = coref_evaluator.evaluate(lines, prefix="", tb_writer=None, global_step=None, official=False)

    logger.info(" >>  Evaluating unified ")
    # for i in range(len(lines)):
    #     for k in ['gold_clusters', 'predicted_clusters']:
    #         lines[i][k] = lines[i][f'original_{k}']
    coref_evaluator = Coref_Evaluator(logger, outdir, experiment_name=f'{split}_unified')
    results = coref_evaluator.evaluate(unified_lines, prefix="", tb_writer=None, global_step=None, official=False)

    logger.info(" >>  Done : ")

    logger.info(str(results))
    

if __name__ == "__main__":
    main()

