import torch

def greedy_cos_idf(
    ref_embedding,
    ref_masks,
    ref_idf,
    hyp_embedding,
    hyp_masks,
    hyp_idf,
):
    """
    Compute greedy matching based on cosine similarity.

    Args:
       'ref_embedding' (torch.Tensor): embeddings of reference sentences, BxKxd, B: batch size, K: longest length, d: bert dimenison
       'ref_lens' (list of int): list of reference sentence length.
       'ref_masks' (torch.LongTensor): BxKxK, BERT attention mask for reference sentences.
       'ref_idf' (torch.Tensor): BxK, idf score of each word piece in the reference sentece
       'hyp_embedding' (torch.Tensor): embeddings of candidate sentences, BxKxd, B: batch size, K: longest length, d: bert dimenison
       'hyp_lens' (list of int): list of candidate sentence length.
       'hyp_masks' (torch.LongTensor): BxKxK, BERT attention mask for candidate sentences.
       'hyp_idf' (torch.Tensor): BxK, idf score of each word piece in the candidate sentence

    Returns
    A tensor that contains the F1 Bertscore for each example in the batch.
    """

    return F_bertscore