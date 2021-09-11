import json
import os
import logging
import random
from collections import OrderedDict, defaultdict
import numpy as np
import torch
from coref_bucket_batch_sampler import BucketBatchSampler
from metrics import CorefEvaluator, MentionEvaluator
from utils.utils import extract_clusters, extract_mentions_to_predicted_clusters_from_clusters, extract_clusters_for_decode
from conll import evaluate_conll

def nested_to_tuple(l):
    if isinstance(l, list):
        for i in range(len(l)):
            l[i] = nested_to_tuple(l[i])
        l = tuple(l)
    return l

class Evaluator:
    def __init__(self, logger, eval_output_dir, experiment_name=''):
        self.eval_output_dir = eval_output_dir
        self.experiment_name = experiment_name
        self.logger = logger
    def evaluate(self, outputs, prefix="", tb_writer=None, global_step=None, official=False):
        assert not official

        post_pruning_mention_evaluator = MentionEvaluator()
        mention_evaluator = MentionEvaluator()
        coref_evaluator = CorefEvaluator()
        losses = defaultdict(list)
        doc_to_prediction = {}
        doc_to_subtoken_map = {}
        for output in outputs:
            # gold_clusters: List[List[List[int]]]
            # predicted_clusters: List[List[List[int]]]
            gold_clusters = nested_to_tuple(output['gold_clusters'])
            predicted_clusters = nested_to_tuple(output['predicted_clusters'])
            doc_key = output['doc_key']

            mention_to_gold_clusters = extract_mentions_to_predicted_clusters_from_clusters(gold_clusters)
            gold_mentions = list(mention_to_gold_clusters.keys())

            # starts, end_offsets, coref_logits, mention_logits = output[-4:]

            # max_antecedents = np.argmax(coref_logits, axis=1).tolist()
            # mention_to_antecedent = {((int(start), int(end)), (int(starts[max_antecedent]), int(end_offsets[max_antecedent]))) for start, end, max_antecedent in
            #                             zip(starts, end_offsets, max_antecedents) if max_antecedent < len(starts)}

            # predicted_clusters, _ = extract_clusters_for_decode(mention_to_antecedent)
            # candidate_mentions = list(zip(starts, end_offsets))

            mention_to_predicted_clusters = extract_mentions_to_predicted_clusters_from_clusters(predicted_clusters)
            predicted_mentions = list(mention_to_predicted_clusters.keys())
            # post_pruning_mention_evaluator.update(candidate_mentions, gold_mentions)
            mention_evaluator.update(predicted_mentions, gold_mentions)
            coref_evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted_clusters,
                                    mention_to_gold_clusters)
            doc_to_prediction[doc_key] = predicted_clusters
            doc_to_subtoken_map[doc_key] = None

        post_pruning_mention_precision, post_pruning_mentions_recall, post_pruning_mention_f1 = post_pruning_mention_evaluator.get_prf()
        mention_precision, mentions_recall, mention_f1 = mention_evaluator.get_prf()
        prec, rec, f1 = coref_evaluator.get_prf()

        # muc, b_cubed, ceafe
        results = []
        for t, (_prec, _rec, _f1) in zip(('muc', 'b_cubed', 'ceafe') , coref_evaluator.get_prf_sep()):
            results.append((f'{t}_prec', _prec))
            results.append((f'{t}_rec', _rec))
            results.append((f'{t}_f1', _f1))

        results += [(key, sum(val) / len(val)) for key, val in losses.items()]
        results += [
            ("post pruning mention precision", post_pruning_mention_precision),
            ("post pruning mention recall", post_pruning_mentions_recall),
            ("post pruning mention f1", post_pruning_mention_f1),
            ("mention precision", mention_precision),
            ("mention recall", mentions_recall),
            ("mention f1", mention_f1),
            ("precision", prec),
            ("recall", rec),
            ("f1", f1)
        ]
        self.logger.info("***** Eval results {} *****".format(prefix))
        for key, values in results:
            if isinstance(values, float):
                self.logger.info(f"  {key} = {values:.3f}")
            else:
                self.logger.info(f"  {key} = {values}")
            if tb_writer is not None and global_step is not None:
                tb_writer.add_scalar(key, values, global_step)

        if self.eval_output_dir:
            output_eval_file = os.path.join(self.eval_output_dir, "eval_results.txt")
            with open(output_eval_file, "a") as writer:
                if prefix:
                    writer.write(f'\n{prefix}:\n')
                for key, values in results:
                    if isinstance(values, float):
                        writer.write(f"{key} = {values:.3f}\n")
                    else:
                        writer.write(f"{key} = {values}\n")

        results = OrderedDict(results)
        results["experiment_name"] = self.experiment_name
        results["data"] = prefix
        with open(os.path.join(self.eval_output_dir, "results.jsonl"), "a+") as f:
            f.write(json.dumps(results) + '\n')

        if official:
            with open(os.path.join(self.args.output_dir, "preds.jsonl"), "w") as f:
                f.write(json.dumps(doc_to_prediction) + '\n')
                f.write(json.dumps(doc_to_subtoken_map) + '\n')

            if self.args.conll_path_for_eval is not None:
                conll_results = evaluate_conll(self.args.conll_path_for_eval, doc_to_prediction, doc_to_subtoken_map)
                official_f1 = sum(results["f"] for results in conll_results.values()) / len(conll_results)
                self.logger.info('Official avg F1: %.4f' % official_f1)

        return results
