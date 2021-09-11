#!/usr/bin/env python
# encoding: utf-8
from __future__ import unicode_literals
import platform
import warnings
from fuzzysearch import find_near_matches
try:
    from fuzzywuzzy.StringMatcher import StringMatcher as SequenceMatcher
except ImportError:
    if platform.python_implementation() != "PyPy":
        warnings.warn('Using slow pure-python SequenceMatcher. Install python-Levenshtein to remove this warning')
    from difflib import SequenceMatcher

from fuzzywuzzy import utils
import functools

def check_for_equivalence(func):
    @functools.wraps(func)
    def decorator(*args, **kwargs):
        if args[0] == args[1]:
            return [0, len(args[0])]
        return func(*args, **kwargs)
    return decorator
def check_for_none(func):
    @functools.wraps(func)
    def decorator(*args, **kwargs):
        if args[0] is None or args[1] is None:
            return []
        return func(*args, **kwargs)
    return decorator


def check_empty_string(func):
    @functools.wraps(func)
    def decorator(*args, **kwargs):
        if len(args[0]) == 0 or len(args[1]) == 0:
            return []
        return func(*args, **kwargs)
    return decorator

###########################
# Basic Scoring Functions #
###########################

@utils.check_for_none
@utils.check_for_equivalence
@utils.check_empty_string
def ratio(s1, s2):
    s1, s2 = utils.make_type_consistent(s1, s2)

    m = SequenceMatcher(None, s1, s2)
    return utils.intr(100 * m.ratio())


@check_for_none
@check_for_equivalence
@check_empty_string
def partial_ratio_with_indices(s1, s2, threshold=0.7, OLD = False):
    """"Return the ratio of the most similar substring
    as a number between 0 and 100."""
    s1, s2 = utils.make_type_consistent(s1, s2)

    if len(s1) <= len(s2):
        shorter = s1
        longer = s2
    else:
        shorter = s2
        longer = s1


    if OLD:
        m = SequenceMatcher(None, shorter, longer)
        blocks = m.get_matching_blocks()

        # each block represents a sequence of matching characters in a string
        # of the form (idx_1, idx_2, len)
        # the best partial match will block align with at least one of those blocks
        #   e.g. shorter = "abcd", longer = XXXbcdeEEE
        #   block = (1,3,3)
        #   best score === ratio("abcd", "Xbcd")
    
    else:
        blocks = find_near_matches(shorter, longer, max_l_dist=min(len(shorter) // 3, 5))
    scores = []
    for block in blocks:
        if OLD:
            long_start = block[1] - block[0] if (block[1] - block[0]) > 0 else 0
            long_end = long_start + len(shorter)
        else:
            long_start, long_end = block.start, block.end

        long_substr = longer[long_start:long_end]
        m2 = SequenceMatcher(None, shorter, long_substr)
        r = m2.ratio()

        scores.append((long_start,long_end,r))

    return list(map(lambda x: x[:2], sorted(scores, key=lambda x: -x[-1])))

