from fuzzywuzzy import fuzz
try:
    from .myfuzz import partial_ratio_with_indices
except:
    from myfuzz import partial_ratio_with_indices
import json
from functools import partial
import functools

from fuzzysearch import find_near_matches


## General
def str_eq(str1, str2, threshold):

    assert isinstance(str1, str), f"str1 type in {type(str1)}"
    assert isinstance(str2, str), f"str2 type in {type(str2)}"

    res =  fuzz.ratio(str1, str2) 
    if threshold <= 1:
        res = res / 100
    # print(str1 , str2, res) # ELAD - REMOVE
    return res >= threshold
 
def clean_str(s, clean_spaces = False):
    s = s.lower()
    if clean_spaces:
        return ''.join(filter(str.isalnum, s))
    else:
        s = " ".join(s.split()) # remove redundant spaces
        return ''.join(filter(lambda x: str.isalnum(x) or x == ' ', s))

def clean_str_list(s, remove_words=False):
    s = list(map(partial(clean_str, clean_spaces=True), s))
    if remove_words:
        s = [x for x in s if len(x)]
    return s

def str_list_isin_NEW(str_list, a, threshold=0.7, max_dups=2):
    """
    Args:
        str: str
        str_list: List[str]
    Return:
        List[int]: [[start: end]]
    """
    if len(str_list) == 0: return []
    if len(a) == 0: return []

    a = clean_str(a)
    # str_list = list(map(clean_str, str_list))
    str_list = clean_str_list(str_list, remove_words=False)
    letter2word = [i for i, wlen in enumerate([len(w) for w in str_list]) for _ in range(wlen + 1)][1:]
    long_str = " ".join(str_list)

    if len(long_str) <= len(a):
        if str_eq(a, long_str, threshold=threshold):
            return [[0, len(str_list)]]
        else:
            return []
        
    slices = partial_ratio_with_indices(a, long_str, threshold)
    assert isinstance(slices, (list, )), f'a: {a}, long_str: {long_str}, threshold: {threshold}, slices: {slices}, type(slices): {type(slices)}'
    # slices = slices[:max_dups]

    # return slices
    # convert to list indices
    try:
        slices = [ ( letter2word[x], letter2word[(y-1) if y < len(long_str) else -1] + 1 ) for x, y in slices]
    except Exception as e:
        print(f"!!!!!!!!!!!")
        print(str_list)
        print(a)
        raise e

    return list(set(slices))

def str_list_isin_OLD(main, sub):
    """
    List[str]
    """
    if len(sub) == 0: return []
    if len(main) == 0: return []
        
    starts = [i for i in range(len(main)) if clean_str(" ".join(main[i:])).startswith(sub)]
    ends = [i for i in range(len(main)) if clean_str(" ".join(main[:i])).endswith(sub)]

    return list(zip(starts, ends))

def str_list_isin(main, sub, threshold = 0.8, args = None):
    indices1 = str_list_isin_OLD(main, sub)
    indices2 = str_list_isin_NEW(main, sub, threshold=threshold - 0.1, max_dups=2)

    indices = indices1 + indices2
    main = clean_str_list(main)
    sub = clean_str(sub)

    indices_new = []
    for x, y in indices:
        while x < len(main) and main[x] == '': x += 1
        while y > 0 and main[y-1] == '': y -= 1
        x = min(x , len(main) - 1)
        y = max(y, 0)
        if x < y:
            indices_new.append((x,y))
    indices = indices_new
    indices = [(x,y, fuzz.ratio(clean_str(" ".join(main[x:y])), sub)) for x,y in indices if str_eq(clean_str(" ".join(main[x:y])), sub, threshold=threshold)]

    return indices

## Base Augmentor

def rec_apply(x, f):
    if isinstance(x, (list, tuple)):
        x = list(map(lambda t: rec_apply(t,f), x))
    else:
        x = f(x)
    return x

def rec2tuple(x):
    if isinstance(x, (list, tuple)):
        x = tuple([rec2tuple(xx) for xx in x])
    return x

def _is_overlap(v, vv):
    return v[0] <= vv[0] and v[1] >= vv[1]

def is_overlap(v, vv):
    return _is_overlap(v, vv) or _is_overlap(vv, v)

def remove_overlapping_intervals(x):
    return tuple([remove_overlapping_intervals_single(xx) for xx in x])

def remove_overlapping_intervals_single(x):
    """
    x: List[List[int]]
    """
    x = list(set(rec2tuple(x)))
    out = set()

    for v in x:
        overlaps = [v]
        for vv in x:
            if is_overlap(v, vv):
                overlaps.append(vv)
        out.add(max(overlaps, key=lambda t: t[1] - t[0]))
    return tuple(sorted(list(out)))




##
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
from statistics import mode
def get_alignment(seq1, seq2, verbose=False):
    """
    Args:
        seq1, seq2: str
    Return:
        List[int]

    Align two strings

    test example:
        seq1 = 'hey, how are you today??? ?'
        seq2 = 'hey, how are <cls> <c> llf kaki you today??? ?'

        ret = get_alignment(seq1,seq2 )
        A = seq1.split()
        B = seq2.split()
        list(zip(B, [A[i] if i>=0 else 'None' for i in ret]))
    OUTPUT:
        [('hey,', 'hey,'),
        ('how', 'how'),
        ('are', 'are'),
        ('<cls>', 'None'),
        ('<c>', 'None'),
        ('llf', 'None'),
        ('kaki', 'None'),
        ('you', 'you'),
        ('today???', 'today???'),
        ('?', '?')]
    """
    # seq1 = " ".join(words)
    # seq2 = 'hello dsa ddsada' + seq1.replace("c", "d d 0") + " dsad dasdd dddd d"

    alignments = pairwise2.align.globalxx(list(seq1), list(seq2), gap_char=['-'])

    # alignments = pairwise2.align.globalxx(seq1, seq2)

    # if verbose:
    #     for alignment in alignments: print(format_alignment(*alignment))
    ret = alignments
    empty_char = '-'
    seqA = ret[0].seqA
    seqB = ret[0].seqB
    assert len(seqA) == len(seqB)

    word_idx_A, word_idx_B = 0, 0
    letter2wordA, letter2wordB = [], []
    newA, newB = [], []
    for i in range(len(seqA)):
        #
        wA, wB = seqA[i], seqB[i]

        # update letter2word
        letter2wordA.append(word_idx_A if wA != empty_char else -1)
        letter2wordB.append(word_idx_B if wB != empty_char else -1)
        if wA == ' ': word_idx_A += 1
        if wB == ' ': word_idx_B += 1

        #
        if wA == empty_char and wB == ' ':
            wA = ' '
        elif wB == empty_char and wA == ' ':
            wB = ' '
        newA.append(wA)
        newB.append(wB)

    newA, newB = map(lambda x: "".join(x).split(), (newA, newB))

    word2inpA, word2inpB = [], []
    cidxA, cidxB = 0, 0

    res = [ -1 for _ in range(len(seq2.split())) ]
    for i, (wordA, wordB) in enumerate(zip(newA, newB)):
        cidx_endA  = cidxA + len(wordA)
        word_idxA = mode(letter2wordA[cidxA:cidx_endA])
        word2inpA.append(word_idxA)
        cidxA = cidx_endA + 1 # len(word) + space

        cidx_endB  = cidxB + len(wordB)
        word_idxB = mode(letter2wordB[cidxB:cidx_endB])
        word2inpB.append(word_idxB)
        cidxB = cidx_endB + 1 # len(word) + space

        res[word_idxB] = word_idxA
    return res


## 


from alignment.sequence import Sequence
from alignment.vocabulary import Vocabulary
from alignment.sequencealigner import SimpleScoring, GlobalSequenceAligner

def seq_aligment(out, inp):
    """
    Args:
        out: str
        inp: List[str]
    """
    out = clean_str_list(out.split(" "))
    inp = clean_str_list(inp)


    # Create sequences to be aligned.
    a = Sequence(out)
    b = Sequence(inp)

    # Create a vocabulary and encode the sequences.
    v = Vocabulary()
    aEncoded = v.encodeSequence(a)
    bEncoded = v.encodeSequence(b)

    # Create a scoring and align the sequences using global aligner.
    scoring = SimpleScoring(2, -1)
    aligner = GlobalSequenceAligner(scoring, -2)
    score, encodeds = aligner.align(aEncoded, bEncoded, backtrace=True)

    encoded = encodeds[0]
    alignment = v.decodeSequenceAlignment(encoded)
    inp, out = alignment.second, alignment.first
    return inp, out

#

def flatten_list_of_lists(lst):
    return [elem for sublst in lst for elem in sublst]


def parse_jsonlines(file_path):
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

###
def is_stopword(x):
    return clean_str(x) in STOP_WORDS
STOP_WORDS= ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own','can', 'will', 'just',]