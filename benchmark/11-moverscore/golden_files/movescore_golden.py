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
    A tensor that contains the moverscore for each example in the batch.
    """

    preds = []
    remove_subwords = True
    n_gram = 1
    stop_words = []
    ref_embedding.div_(torch.norm(ref_embedding, dim=-1).unsqueeze(-1))
    hyp_embedding.div_(torch.norm(hyp_embedding, dim=-1).unsqueeze(-1))

    ref_embedding_max, _ = torch.max(ref_embedding[-5:], dim=0, out=None)
    hyp_embedding_max, _ = torch.max(hyp_embedding[-5:], dim=0, out=None)

    ref_embedding_min, _ = torch.min(ref_embedding[-5:], dim=0, out=None)
    hyp_embedding_min, _ = torch.min(hyp_embedding[-5:], dim=0, out=None)

    ref_embedding_avg = ref_embedding[-5:].mean(0)
    hyp_embedding_avg = hyp_embedding[-5:].mean(0)

    ref_embedding = torch.cat([ref_embedding_min, ref_embedding_avg, ref_embedding_max], -1)
    hyp_embedding = torch.cat([hyp_embedding_min, hyp_embedding_avg, hyp_embedding_max], -1)

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

        ids = ref_ids
        embedding = ref_embedding[i]
        idf = ref_idf[i]
        n = n_gram
        o = 1
        new_a = []
        new_idf = []
        a = np.array(ids)
        w = n
        o = o
        if a.size - w + 1 <= 0:
            w = a.size
        sh = (a.size - w + 1, w)
        st = a.strides * 2
        view = np.lib.stride_tricks.as_strided(a, strides=st, shape=sh)[0::o]
        slide_wins = view.copy().tolist()

        for slide_win in slide_wins:
            new_idf.append(idf[slide_win].sum().item())
            scale = idf[slide_win] / (idf[slide_win].sum(0) + 0.00001)
            scale = scale.unsqueeze(-1)
            tmp = (scale * embedding[slide_win]).sum(0)
            new_a.append(tmp)
        new_a = torch.stack(new_a, 0)
        ref_embedding_i = new_a
        ref_idf_i = new_idf

        ids = hyp_ids
        embedding = hyp_embedding[i]
        idf = hyp_idf[i]
        n = n_gram
        o = 1

        new_a = []
        new_idf = []
        a = np.array(ids)
        w = n
        o = o
        if a.size - w + 1 <= 0:
            w = a.size
        sh = (a.size - w + 1, w)
        st = a.strides * 2
        view = np.lib.stride_tricks.as_strided(a, strides=st, shape=sh)[0::o]
        slide_wins = view.copy().tolist()

        for slide_win in slide_wins:
            new_idf.append(idf[slide_win].sum().item())
            scale = idf[slide_win] / (idf[slide_win].sum(0) + 0.00001)
            scale = scale.unsqueeze(-1)
            tmp = (scale * embedding[slide_win]).sum(0)
            new_a.append(tmp)
        new_a = torch.stack(new_a, 0)
        hyp_embedding_i = new_a
        hyp_idf_i = new_idf

        raw = torch.cat([ref_embedding_i, hyp_embedding_i], 0)
        raw.div_(torch.norm(raw, dim=-1).unsqueeze(-1) + 0.000001)

        x = raw
        y = raw
        x_norm = (x ** 2).sum(1).view(-1, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
        y_t = torch.transpose(y, 0, 1)
        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        distance_matrix = torch.clamp(dist, 0.0, np.inf)

        c1 = np.zeros(len(ref_idf_i) + len(hyp_idf_i), dtype=np.double)
        c2 = np.zeros(len(ref_idf_i) + len(hyp_idf_i), dtype=np.double)

        c1[:len(ref_idf_i)] = ref_idf_i
        c2[-len(hyp_idf_i):] = hyp_idf_i

        c1 = c1 / (np.sum(c1) + 0.00001)
        c2 = c2 / (np.sum(c2) + 0.00001)
        score = 1 - emd(c1, c2, distance_matrix.double().cpu().numpy())
        preds.append(score)
    return preds
