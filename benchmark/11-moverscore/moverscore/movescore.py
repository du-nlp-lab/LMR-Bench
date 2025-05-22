import torch
import numpy as np
import string
from pyemd import emd

def word_mover_score(ref_embedding, ref_idf, ref_tokens, hyp_embedding, hyp_idf, hyp_tokens):
    """
    Args:
       'ref_embedding' (torch.Tensor): embeddings of reference sentences, 12XBxKxd, B: batch size, K: longest length, d: bert dimenison
       'ref_idf' (torch.Tensor): BxK, idf score of each word piece in the reference sentence
        ref_tokens' (list of list): BxK, tokens of the reference sentence
       'hyp_embedding' (torch.Tensor): embeddings of candidate sentences, 12XBxKxd, B: batch size, K: longest length, d: bert dimenison
       'hyp_idf' (torch.Tensor): BxK, idf score of each word piece in the candidate sentence
       'hyp_tokens' (list of list): BxK, tokens of the candidate sentence

    Returns
    A list that contains the moverscore for each example in the batch.
    """
    preds = []
    remove_subwords = True
    n_gram = 1
    stop_words = []
    for i in range(len(ref_tokens)):
        if remove_subwords:
            ref_ids = [k for k, w in enumerate(ref_tokens[i]) if
                       w not in set(string.punctuation) and '##' not in w and w not in stop_words]
            hyp_ids = [k for k, w in enumerate(hyp_tokens[i]) if
                       w not in set(string.punctuation) and '##' not in w and w not in stop_words]
        else:
            ref_ids = [k for k, w in enumerate(ref_tokens[i]) if
                       w not in set(string.punctuation) and w not in stop_words]
            hyp_ids = [k for k, w in enumerate(hyp_tokens[i]) if
                       w not in set(string.punctuation) and w not in stop_words]


    return preds
