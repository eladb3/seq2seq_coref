import json
import os
from utils import distributed as du
from collections import Counter, defaultdict
import random
from .utils import *
import pickle

class FakeLogger():
    def info(self, p):
        print(p)
class AugmentorBase:
    def __init__(self, logger, args = None, add_speakers = True, debug=False, outfile_name = '', base = '/home/yandex/AMNLP2021/eladb3/coref', max_sentences=10):
        self.splits=['train', 'dev', 'test']
        self.args = args
        self.add_speakers = add_speakers
        self.debug = debug >= 1
        self.hard_debug = debug >= 2
        self.logger = logger or FakeLogger()
        self.base = base
        self.outfile_name = outfile_name
        self.max_sentences = max_sentences
        self.const = {
                      'speaker_start': '<speaker_start>',
                      'speaker_end': '<speaker_end>',
                    }

    def prepare_all_splits(self):
        base = self.base
        ret = []
        for split in self.splits:
            ret.append(self.augment(split))
        return ret


    def get_examples(self, split, mkdir=False, max_sentences = None):
        if max_sentences is None:
            max_sentences = self.max_sentences
        out_path, file_path = self.get_outpath(split, mkdir=mkdir)
        examples, max_mention_num, max_cluster_size, max_num_clusters = self._parse_jsonlines(file_path)
        if max_sentences > 0:
            examples = split2sentences(examples, max_sentences=max_sentences)
        return examples, max_mention_num, max_cluster_size, max_num_clusters

    def augment(self, split):
        assert split in ['train', 'dev', 'test']

        out_path, file_path = self.get_outpath(split, mkdir=True)

        if False: #os.path.isfile(out_path):
            self.logger.info(f'{out_path} already exists!')
        else:
            if du.is_root_proc():
                self.logger.info(f'Start generating {out_path} ... ')
                examples, max_mention_num, max_cluster_size, max_num_clusters = self.get_examples(split, mkdir=True)
                augmented_data = self.cor2seq(examples)

                # save to jsonlines
                with open(out_path, 'wt') as f:
                    f.writelines([f'{json.dumps(t)}\n' for t in augmented_data])
        du.synchronize()
        return out_path
    
    def unaugment(self, split, predictions):
        """
        predictions: [[{"idx":int, prediction":str, "label": str}]
        return:
            [{doc_key:str, gold_clusters:List[List[List[int]]], predicted_clusters:List[List[List[int]]], prediction:str, label:str}]
        """
        # 
        assert split in ['train', 'dev', 'test']
        out_path, file_path = self.get_outpath(split, mkdir=False)
        assert all(os.path.isfile(f) for f in [out_path, file_path])
        examples, max_mention_num, max_cluster_size, max_num_clusters = self.get_examples(split, mkdir=False)
        orig_examples, max_mention_num, max_cluster_size, max_num_clusters = self.get_examples(split, mkdir=False, max_sentences = 0)
        orig_examples = {e[0]:e for e in orig_examples}
        lines = self._read_jsonlines_outpath(out_path)

        # first match to docs
        preds = {l['idx']:l for l in predictions}
        preds_orig = {}
        n_found = 0
        for idx, ((doc_key, words, clusters, speakers), augemented) in enumerate(zip(examples, lines)):
            if idx % 10 == 0:
                self.logger.info(f"{idx} / {len(examples)}")
            k = idx
            assert augemented['doc_key'] == doc_key
            if k in preds:
                assert 'doc_key' not in preds[k]
                n_found += 1
                preds[k]['doc_key'] = doc_key
                preds[k]['augemented_label'] = augemented['cor']
                preds[k]['augemented_txt'] = augemented['text']
                preds[k]['words'] = words
                preds[k]['gold_clusters'] = clusters
                preds[k]['speakers'] = speakers
                if self.debug and (not self.hard_debug):
                    preds[k]['predicted_clusters'] = self.seq2cor_single(doc_key, words, clusters, speakers, augemented, preds[k]['augemented_label'], preds[k]['augemented_label'])
                elif self.hard_debug:
                    preds[k]['predicted_clusters'] = clusters
                elif self.args.empty_debug:
                    preds[k]['predicted_clusters'] = []
                else:
                    preds[k]['predicted_clusters'] = self.seq2cor_single(doc_key, words, clusters, speakers, augemented, preds[k]['prediction'], preds[k]['augemented_label'])
                preds[k]['predicted_clusters'] = remove_overlapping_intervals(preds[k]['predicted_clusters'])
                orig_doc_key, split_idx, start, end = split_doc_key(doc_key)
                # preds[k]['original_gold_clusters'] = orig_examples[orig_doc_key][2]
                # preds[k]['original_predicted_clusters'] = rec_apply(preds[k]['predicted_clusters'], f = lambda x: x + start)
                if orig_doc_key not in preds_orig:
                    preds_orig[orig_doc_key] = {'doc_key': orig_doc_key, 'gold_clusters': orig_examples[orig_doc_key][2], 'predicted_clusters':[]}
                preds_orig[orig_doc_key]['predicted_clusters'].extend(rec_apply(preds[k]['predicted_clusters'], f = lambda x: x + start))


        assert n_found == len(preds)

        return list(preds.values()), list(preds_orig.values())

        

    def _parse_jsonlines(self, file_path):
            examples = []
            max_mention_num = -1
            max_cluster_size = -1
            max_num_clusters = -1
            with open(file_path, 'r') as f:
                for line in f:
                    d = json.loads(line.strip())
                    doc_key = d["doc_key"]
                    input_words = flatten_list_of_lists(d["sentences"])
                    clusters = d["clusters"]
                    max_mention_num = max(max_mention_num, len(flatten_list_of_lists(clusters)))
                    max_cluster_size = max(max_cluster_size, max(len(cluster) for cluster in clusters) if clusters else 0)
                    max_num_clusters = max(max_num_clusters, len(clusters) if clusters else 0)
                    speakers = flatten_list_of_lists(d["speakers"])
                    examples.append((doc_key, input_words, clusters, speakers))
            return examples, max_mention_num, max_cluster_size, max_num_clusters

    def _read_jsonlines_outpath(self, out_path):
        with open(out_path, 'rt') as f:
            lines = [json.loads(l) for l in f.read().split('\n') if l]
        return lines

    def cor2seq(self, examples):
        ret = []
        for idx, sample in enumerate(examples):
            ret.append(self.cor2seq_single(*sample))
        return ret

    def get_outpath(self, split,mkdir=False):
        assert split in ['train', 'dev', 'test']
        outfile_name, base = self.outfile_name, self.base
        file_path = os.path.join(base, f'{split}.english.jsonlines')

        out_dir = os.path.join(base, 'augmented_data')
        if mkdir: os.makedirs(out_dir, exist_ok=True, mode=0o777)
        if outfile_name: outfile_name = outfile_name + '_'
        name = f'{outfile_name}{split}_{self.max_sentences}_{"with_speakers" if self.add_speakers else "without_speakers"}'
        if getattr(self, 'add_loc_tokens', False):
            name = f'{name}_witLocTokens'
        name = f'{name}_{type(self).__name__}.json'
        out_path = os.path.join(out_dir, name)

        return out_path, file_path


    def cor2seq_single(self, doc_key, words, clusters, speakers):
        raise NotImplementedError("Please Implement this method")

    def get_max_len(self, tokenizer, split='train'):
        txt_max = 0
        sum_max = 0
        outpath, _ = self.get_outpath(split)
        assert os.path.isfile(outpath)
        with open(outpath, 'rt') as f: lines = f.read().split('\n')[:-1]
        for l in lines:
            d = json.loads(l)
            txt_max = max(txt_max, len(tokenizer.encode(d['text'])))
            sum_max = max(sum_max, len(tokenizer.encode(d['cor'])))
        return txt_max, sum_max

    def prepare_inputs(self, words, speakers, SPEAKER_START, SPEAKER_END):
        if self.add_speakers:
            return prepare_inputs(words, speakers, SPEAKER_START, SPEAKER_END)
        return words

## Utils

def prepare_inputs(words, speakers, SPEAKER_START, SPEAKER_END):
    token_ids = []
    last_speaker = None

    for idx, (word, speaker) in enumerate(zip(words, speakers)):

        if last_speaker != speaker:
            speaker_prefix = [SPEAKER_START] + [" " + speaker] + [SPEAKER_END]
            last_speaker = speaker
        else:
            speaker_prefix = []

        token_ids.extend(speaker_prefix)
        tokenized = [" " + word]
        token_ids.extend(tokenized)
    return token_ids

class debug_augmentor(AugmentorBase):
    def __init__(self, logger, **kwargs):
        super().__init__(logger, **kwargs)
        self.const = {}

    def cor2seq_single(self, doc_key, words, clusters, speakers):
        """
        Args:
            doc_key: str
            words: List[str]
            clusters: List[List[[x,y]]]
            speakers: List[str]
        Out:
            {'text': str, 'cor': str}
        """
        text = words
        cor = list(reversed(text))

        out = {'text':text, 'cor':cor, 'doc_key':doc_key}
        out = {k:" ".join(v) for k,v in out.items()}
        return out


def split_doc_key(doc_key):
    doc_key, s = doc_key.split('_split_')
    split_idx, _, start, _, end= s.split('_')
    split_idx, start, end = map(int, (split_idx, start, end))
    return doc_key, split_idx, start, end
def split2sentences(examples, max_sentences=10):
    examples_out = []
    for idx, (doc_key, input_words, clusters, speakers) in enumerate(examples):

        end_indices = [i for i, word in enumerate(input_words) if word == '.']
        start_indices = [-1] + end_indices[:-1]
        indices = list(zip(start_indices, end_indices))
        split_indices = [indices[i:i + max_sentences] for i in list(range(0,len(indices),max_sentences))]
        for split_idx, split  in enumerate(split_indices):
            start, end = zip(*split)
            start, end = min(start), max(end)
            start += 1 # start after '.'
            end += 1 # include '.' in sentence
            _input_words = input_words[start:end]
            _doc_key = f'{doc_key}_split_{split_idx}_start_{start}_end_{end}'
            _speakers = speakers[start:end]
            _clusters = filter_clusters(clusters, start, end)
            examples_out.append((_doc_key, _input_words, _clusters, _speakers))
    return examples_out

def filter_clusters(clusters, start, end):
    ret = []
    for cluster in clusters:
        new_cluster = []
        length = len(range(start,end))
        for x,y in cluster:
            if x not in range(start, end): continue
            x, y = map(lambda t: min(t-start, length-1), (x,y))
            new_cluster.append([x, y])
        if len(new_cluster) > 1:
            ret.append(new_cluster)
    return ret

##
class naive_augmentor(AugmentorBase):
    def __init__(self, logger, **kwargs):
        super().__init__(logger, **kwargs)

        self.add_loc_tokens = self.args.naive_augmentor_add_loc_tokens
        self.replace_loc_tokens = self.args.replace_loc_tokens
        self.threshold = self.args.naive_augmentor_threshold
        
        self.const.update({'cluster_sep': '<cluster_sep>',
                        'ref_sep': '<ref_sep>',})
        if True: #self.add_loc_tokens:
            self.const.update({'start_loc':'<start_loc>', 'end_loc':'<end_loc>'})

    def cor2seq_single(self, doc_key, words, clusters, speakers):
        """
        Args:
            doc_key: str
            words: List[str]
            clusters: List[List[[x,y]]]
            speakers: List[str]
        Out:
            {'text': str, 'cor': str}
        """
        text = words
        cor = []

        if self.add_loc_tokens:
            text = self._add_loc_tokens(text)

        # LABELS
        for cluster in clusters:
            cor.append(self.const['cluster_sep'])
            for i, (x,y) in enumerate(cluster):
                a =  [text[i] for i in range(x,y+1)]
                if i < len(cluster) - 1:
                    a.append(self.const['ref_sep'])
                cor.extend(a)
        cor.append(self.const['cluster_sep'])
        inputs = self.prepare_inputs(text, speakers, self.const['speaker_start'], self.const['speaker_end'])
        out = {'text':inputs, 'cor':cor, 'doc_key':doc_key}
        out = {k:" ".join(v) if k in ['text', 'cor'] else v for k,v in out.items()}
        return out

    def seq2cor_single(self, doc_key, words, clusters, speakers, augemented, prediction, label):
        """
        return:
            predicted_clusters: List[List[List[int]]]
        """
        if self.add_loc_tokens:
            words = self._add_loc_tokens(words)
            if self.replace_loc_tokens:
                words = self._prepare_loc_tokens(words)
                prediction = self._prepare_loc_tokens(prediction)

        clusters = [cluster.split(self.const['ref_sep']) for cluster in prediction.split(self.const['cluster_sep']) if cluster]
        for i in range(len(clusters)):
            cluster = clusters[i]
            for iref in range(len(cluster)):
                ref = cluster[iref]
                ref = clean_str(ref)
                # print(f"Before:'{cluster[iref]}', After: '{ref}'")
                cluster[iref] = ref
            clusters[i] = cluster
    
        words = [clean_str(w) for w in words]
        predicted_clusters = []

        for cluster in clusters:
            pcluster= []
            cluster_scores = {}
            for ref in cluster:
                idxs =  str_list_isin(words, ref, threshold=self.threshold)
                # print(f"{ref}: {idxs}")
                for i, j, score in idxs:
                    if (i,j-1) not in pcluster:
                        pcluster.append((i,j-1))
                        cluster_scores[(i,j-1)] = score
                    else:
                        cluster_scores[(i,j-1)] = max(score, cluster_scores[(i,j-1)])
            if pcluster:
                # remove intersect clusters
                pcluster = remove_intersections(pcluster, cluster_scores)
                predicted_clusters.append(pcluster)
        return predicted_clusters

    def _add_loc_tokens(self, inp):
        """
        Args:
            inp: List[str]
        """
        add_list = ['' for _ in range(len(inp)) ]
        words_indices = defaultdict(list)

        for idx, word in enumerate(inp):
            word = clean_str(word)
            if len(word) == 0: continue
            words_indices[word].append(idx)
        
        words_indices = {k:v for k,v in words_indices.items() if (len(v) > 1 or is_stopword(k))}

        for word, indices in words_indices.items():
            for enum_idx, idx in enumerate(indices):
                add_list[idx] += f"{self.const['start_loc']}{enum_idx}{self.const['end_loc']}"
        
        ret = [loc + word for loc, word in zip(add_list, inp)]
        return ret

    def _prepare_loc_tokens(self, words):
        """
        replace all loc tokens with indices
        """
        if isinstance(words, list):
            for i in ['start', 'end']:
                words = [x.replace(self.const[f'{i}_loc'], '') for x in words]
        elif isinstance(words, str):
            for i in ['start', 'end']:
                words = words.replace(self.const[f'{i}_loc'], '')
        return words


##
class naive_augmentor2(AugmentorBase):
    def __init__(self, logger, **kwargs):
        super().__init__(logger, **kwargs)

        self.add_loc_tokens = True

        self.const.update({'cluster_sep': '<cluster_sep>',
                        'ref_sep': '<ref_sep>',})
        self.MAX_LOC_TOKENS = 10
        if self.add_loc_tokens:
            self.const.update({f'loc_{i}':f'<loc_{i}>' for i in range(self.MAX_LOC_TOKENS)})

    def cor2seq_single(self, doc_key, words, clusters, speakers):
        """
        Args:
            doc_key: str
            words: List[str]
            clusters: List[List[[x,y]]]
            speakers: List[str]
        Out:
            {'text': str, 'cor': str}
        """
        text = words
        cor = []

        if self.add_loc_tokens:
            text = self._add_loc_tokens(text)

        # LABELS
        for cluster in clusters:
            cor.append(self.const['cluster_sep'])
            for i, (x,y) in enumerate(cluster):
                a =  [text[i] for i in range(x,y+1)]
                if i < len(cluster) - 1:
                    a.append(self.const['ref_sep'])
                cor.extend(a)
        cor.append(self.const['cluster_sep'])
        inputs = self.prepare_inputs(text, speakers, self.const['speaker_start'], self.const['speaker_end'])
        out = {'text':inputs, 'cor':cor, 'doc_key':doc_key}
        out = {k:" ".join(v) if k in ['text', 'cor'] else v for k,v in out.items()}
        return out

    def seq2cor_single(self, doc_key, words, clusters, speakers, augemented, prediction, label):
        """
        return:
            predicted_clusters: List[List[List[int]]]
        """
        if self.add_loc_tokens:
            words = self._add_loc_tokens(words)
            words = self._prepare_loc_tokens(words)
            prediction = self._prepare_loc_tokens(prediction)


        clusters = [cluster.split(self.const['ref_sep']) for cluster in prediction.split(self.const['cluster_sep']) if cluster]
        for i in range(len(clusters)):
            cluster = clusters[i]
            for iref in range(len(cluster)):
                ref = cluster[iref]
                ref = clean_str(ref)
                # print(f"Before:'{cluster[iref]}', After: '{ref}'")
                cluster[iref] = ref
            clusters[i] = cluster
    
        words = [clean_str(w) for w in words]
        predicted_clusters = []

        for cluster in clusters:
            pcluster= []
            cluster_scores = {}
            for ref in cluster:
                idxs =  str_list_isin(words, ref)
                # print(f"{ref}: {idxs}")
                for i, j, score in idxs:
                    if (i,j-1) not in pcluster:
                        pcluster.append((i,j-1))
                        cluster_scores[(i,j-1)] = score
                    else:
                        cluster_scores[(i,j-1)] = max(score, cluster_scores[(i,j-1)])
            if pcluster:
                # remove intersect clusters
                pcluster = remove_intersections(pcluster, cluster_scores)
                predicted_clusters.append(pcluster)
        return predicted_clusters

    def _add_loc_tokens(self, inp):
        """
        Args:
            inp: List[str]
        """
        add_list = ['' for _ in range(len(inp)) ]
        words_indices = defaultdict(list)

        for idx, word in enumerate(inp):
            if len(word) < 3: continue
            words_indices[clean_str(word)].append(idx)
        
        words_indices = {k:v for k,v in words_indices.items() if len(v) > 1}

        for word, indices in words_indices.items():
            for enum_idx, idx in enumerate(indices):
                enum_idx = enum_idx % self.MAX_LOC_TOKENS
                add_list[idx] += f"{self.const[f'loc_{enum_idx}']}"
        
        ret = [loc + word for loc, word in zip(add_list, inp)]
        return ret

    def _prepare_loc_tokens(self, words):
        """
        replace all loc tokens with indices
        """
        if isinstance(words, list):
            for i in range(self.MAX_LOC_TOKENS):
                words = [x.replace(self.const[f'loc_{i}'], str(i)) for x in words]
        elif isinstance(words, list):
            for i in range(self.MAX_LOC_TOKENS):
                words = words.replace(self.const[f'loc_{i}'], str(i))
        return words

def getOverlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))
def remove_intersections(preds, scores):
    # preds: [(x,y)]
    # scores: {(x,y) --> score}
    LENGTH_PENALTY = 10
    remove = [False for _ in range(len(preds))]
    for ix, x in enumerate(preds):
        for iy, y in enumerate(preds):
            if x == y or remove[ix] or remove[iy]: continue
            ax, bx = x
            ay, by = y
            bx += 1
            by += 1
            if getOverlap([ax, bx], [ay,by]) > 0:
                scorex = scores[x]
                scorey = scores[y]
                lx = bx - ax
                ly = by - ay
                if lx < ly: scorex -= LENGTH_PENALTY
                elif ly < lx: scorey -= LENGTH_PENALTY
                scorex = (scorex, lx)
                scorey = (scorey, ly)
                if scorex > scorey: remove[iy] = True
                else: remove[ix] = True
    ret = [preds[i] for i in range(len(preds)) if not remove[i]]
    return ret

## Partial

## under construction
##
class start_end_tokens_augmentor(AugmentorBase):
    def __init__(self, logger, **kwargs):
        super().__init__(logger, **kwargs)

        self.max_clusters = self.args.start_end_tokens_augmentor_max_clusters
        self.shuffle_cluster_indices = self.args.start_end_tokens_augmentor_shuffle_cluster_indices
        self.better_alignment = self.args.start_end_tokens_augmentor_better_alignment

        self.const.update({f'cluster_start_{i}': f'<cluster_start_{i}>' for i in range(self.max_clusters)})
        self.const.update({f'cluster_end_{i}': f'<cluster_end_{i}>' for i in range(self.max_clusters)})
        self.const_inv = {v:k for k,v in self.const.items()}


    def cor2seq_single(self, doc_key, words, clusters, speakers):
        """
        Args:
            doc_key: str
            words: List[str]
            clusters: List[List[[x,y]]]
            speakers: List[str]
        Out:
            {'text': str, 'cor': str}
        """
        text = words

        add_text = ["" for _ in range(len(words) + 1)]
        clusters = clusters[:self.max_clusters]
        cluster_indices = list(range(self.max_clusters))
        if self.shuffle_cluster_indices:
            random.shuffle(cluster_indices)
        # LABELS
        for cluster_idx, cluster in enumerate(clusters):
            if cluster_idx >= self.max_clusters: break
            cluster_idx = cluster_indices[cluster_idx]
            for i, (x,y) in enumerate(cluster):
                add_text[x] += f"{self.const[f'cluster_start_{cluster_idx}']} "
                add_text[y+1] += f"{self.const[f'cluster_end_{cluster_idx}']} "
        add_text = [w.rstrip() for w in add_text]
        cor = []
        for x,y in zip(add_text[:-1], text): cor.extend([x,y])
        cor.append(add_text[-1])
        inputs = self.prepare_inputs(words, speakers, self.const['speaker_start'], self.const['speaker_end'])
        out = {'text':inputs, 'cor':cor, 'doc_key':doc_key}
        out = {k:" ".join(v) if k in ['text', 'cor'] else v for k,v in out.items()}
        return out


    def check_if_cluster_const(self, token):
        token = token.replace(" ", "").casefold()
        if token in self.const_inv and 'cluster' in token:
            start = 'start' in token
            end = 'end' in token
            if start:
                cluster_idx = int(token[15:-1])
            else: # end
                cluster_idx = int(token[13:-1])
            assert (start or end) and not (start and end)
            assert cluster_idx < self.max_clusters
            name = 'start' if start else 'end'
            return name, cluster_idx
        else:
            return -1, -1

    def seq2cor_single(self, doc_key, words, clusters, speakers, augemented, prediction, label):
        """
        return:
            predicted_clusters: List[List[List[int]]]
        """
        # remove double spaces
        prediction = " ".join([x for x in prediction.split() if x])
        prediction = prediction.split(" ")

        pred2inp = self.align(words, prediction)
        # pred2inp = dict(zip(*[range(10000) for i in range(2)])) # ELAD REMOVE

        def get_start_idx_input(start_idx_pred):
            for i in range(self.max_clusters):
                if start_idx_pred + i < len(pred2inp) and pred2inp[start_idx_pred + i] != -1:
                    return pred2inp[start_idx_pred + i]
            return -1

        predicted_clusters = [[] for _ in range(self.max_clusters)]
        for idx_pred, pred_word in enumerate(prediction):
            type_, cluster_idx = self.check_if_cluster_const(pred_word)
            if type_ == 'start':
                # fine end
                for idx_pred2, pred_word2 in enumerate(prediction[idx_pred:]):
                    idx_pred2 += idx_pred
                    type_2, cluster_idx2 = self.check_if_cluster_const(pred_word2)
                    if type_2 == 'end' and cluster_idx2 == cluster_idx:
                        # found end
                        predicted_clusters[cluster_idx].append([get_start_idx_input(idx_pred), get_start_idx_input(idx_pred2) - 1])
                        break
        return [x for x in predicted_clusters if x]

    def better_align_wrapper_EXT(self, inp, tar, num_tries = 15):
        cache = "./tmp_files"
        os.makedirs(cache, exist_ok=True, mode=0o777)
        file1 = os.path.join(cache, 'file1.txt')
        file2 = os.path.join(cache, 'file2.txt')
        out = os.path.join(cache, 'out.txt')

        # remote out if exists
        if os.path.isfile(out):
            os.remove(out)

        # write files
        with open(file1, 'wb') as f: pickle.dump(inp, f)
        with open(file2, 'wb') as f: pickle.dump(tar, f)
        assert all([os.path.isfile(x) for x in (file1, file2)]), f"{[os.path.isfile(x) for x in (file1, file2, out)]}"

        for i in range(num_tries):
            os.system(f"python -u DataAugmentation/align_wrapper.py --file1 {file1} --file2 {file2} --out {out}")
            if os.path.isfile(out): break
            if i == num_tries - 1:
                raise Exception(f"Failed to align after {num_tries} tries")
            self.logger.warn(f'Alignment failed after {i} tries')

        with open(out, 'rb') as f:
            res = pickle.load(f)

        assert all([os.path.isfile(x) for x in (file1, file2, out)]), f"{[os.path.isfile(x) for x in (file1, file2, out)]}"

        # remove files
        os.remove(out)
        os.remove(file1)
        os.remove(file2)

        return res



    def better_align_wrapper(self, inp, tar, num_tries = 15):
        return get_alignment(inp, tar)


    def align(self, inp, tar):
        if self.better_alignment:
            tar2inp = self.better_align_wrapper( *map(lambda x: " ".join(x), (inp , tar)) )
        else:
            inp2tar, tar2inp = self._align(inp, tar)
        return tar2inp

    def _align(self, inp, tar):
        inp2tar, tar2inp = defaultdict(lambda: -1), defaultdict(lambda: -1)

        idx_inp, idx_tar = 0, 0
        while idx_inp < len(inp) and idx_tar < len(tar):
            if tar[idx_tar] not in self.const_inv:
                inp2tar[idx_inp] = idx_tar
                tar2inp[idx_tar] = idx_inp
                idx_inp += 1
            idx_tar += 1

        return inp2tar, tar2inp

##
class naive_augmentor_with_loc_tokens(AugmentorBase):
    def __init__(self, logger, **kwargs):
        super().__init__(logger, **kwargs)
        self.const = {'cluster_sep': '<cluster_sep>',
                        'ref_sep': '<ref_sep>',
                        'speaker_start': '<speaker_start>',
                        'speaker_end': '<speaker_end>'}
    def cor2seq_single(self, doc_key, words, clusters, speakers):
        """
        Args:
            doc_key: str
            words: List[str]
            clusters: List[List[[x,y]]]
            speakers: List[str]
        Out:
            {'text': str, 'cor': str}
        """
        text = words
        cor = []

        # LABELS
        for cluster in clusters:
            cor.append(self.const['cluster_sep'])
            for i, (x,y) in enumerate(cluster):
                a =  [text[i] for i in range(x,y+1)]
                if i < len(cluster) - 1:
                    a.append(self.const['ref_sep'])
                cor.extend(a)
        cor.append(self.const['cluster_sep'])
        inputs = prepare_inputs(words, speakers, self.const['speaker_start'], self.const['speaker_end'])
        out = {'text':inputs, 'cor':cor, 'doc_key':doc_key}
        out = {k:" ".join(v) if k in ['text', 'cor'] else v for k,v in out.items()}
        return out

    def seq2cor_single(self, doc_key, words, clusters, speakers, augemented, prediction, label):
        """
        return:
            predicted_clusters: List[List[List[int]]]
        """

        clusters = [cluster.split(self.const['ref_sep']) for cluster in prediction.split(self.const['cluster_sep']) if cluster]
        for i in range(len(clusters)):
            cluster = clusters[i]
            for iref in range(len(cluster)):
                ref = cluster[iref]
                ref = clean_str(ref)
                # print(f"Before:'{cluster[iref]}', After: '{ref}'")
                cluster[iref] = ref
            clusters[i] = cluster
    
        words = [clean_str(w) for w in words]
        predicted_clusters = []

        for cluster in clusters:
            pcluster=[]
            for ref in cluster:
                idxs =  str_list_isin(words, ref)
                # print(f"{ref}: {idxs}")
                for i, j in idxs:
                    if (i,j-1) not in pcluster:
                        pcluster.append((i,j-1))
            if pcluster:
                predicted_clusters.append(pcluster)
        return predicted_clusters

class my_augmentor(AugmentorBase):
    def __init__(self, logger, **kwargs):
        super().__init__(logger, **kwargs)
        self.max_occurences = 10
        self.const = {'cluster_sep': '<cluster_sep>',
                        'ref_sep': '<ref_sep>',
                        'speaker_start': '<speaker_start>',
                        'speaker_end': '<speaker_end>'}
        self.const.update({f'occurence_{i}':f'<occurence_{i}>' for i in range(self.max_occurences)})
    def cor2seq_single(self, doc_key, words, clusters, speakers):
        """
        Args:
            doc_key: str
            words: List[str]
            clusters: List[List[[x,y]]]
            speakers: List[str]
        Out:
            {'text': str, 'cor': str}
        """
        words = self.add_occurences(words)

        cor = []
        # LABELS
        for cluster in clusters:
            cor.append(self.const['cluster_sep'])
            for x,y in cluster:
                cor.extend([self.const['ref_sep']] + [words[i] for i in range(x,y+1)] + [self.const['ref_sep']])
        cor.append(self.const['cluster_sep'])
        inputs = self.prepare_inputs(words, speakers, self.const['speaker_start'], self.const['speaker_end'])
        out = {'text':inputs, 'cor':cor, 'doc_key':doc_key}
        out = {k:" ".join(v) if k in ['text', 'cor'] else v for k,v in out.items()}
        return out

    def add_occurences(self, words):
        c = Counter(words)
        doubles = [x for x in  c.keys() if c[x] > 1 and len(x) > 2]
        c = Counter()
        for i in range(len(words)):
            w = words[i]
            if w in doubles:
                idx = c[w] % self.max_occurences
                idx = self.const[f'occurence_{idx}']
                words[i] = f'{idx}{w}'
            c[w] += 1
        return words

    def prepare_inputs(self, words, speakers, speaker_start, speaker_end):
        words = prepare_inputs(words, speakers, speaker_start, speaker_end)
        return words
        

    def seq2cor_single(self, doc_key, words, clusters, speakers, augemented, prediction, label):
        """
        return:
            predicted_clusters: List[List[List[int]]]
        """
        words = self.add_occurences(words)

        clusters = [cluster.split(self.const['ref_sep']) for cluster in prediction.split(self.const['cluster_sep']) if cluster]
        for i in range(len(clusters)):
            cluster = clusters[i]
            for iref in range(len(cluster)):
                ref = cluster[iref]
                ref = clean_str(ref)
                cluster[iref] = ref
            clusters[i] = cluster
    
        words = [clean_str(w) for w in words]
        predicted_clusters = []

        for cluster in clusters:
            pcluster=[]
            for ref in cluster:
                idxs =  str_list_isin(words, ref)
                for i, j in idxs:
                    if (i,j-1) not in pcluster:
                        pcluster.append((i,j-1))
            if pcluster:
                predicted_clusters.append(pcluster)
        return predicted_clusters


###
