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
       'ref_idf' (torch.Tensor): BxK, idf score of each word piece in the reference setence
       'hyp_embedding' (torch.Tensor): embeddings of candidate sentences, BxKxd, B: batch size, K: longest length, d: bert dimenison
       'hyp_lens' (list of int): list of candidate sentence length.
       'hyp_masks' (torch.LongTensor): BxKxK, BERT attention mask for candidate sentences.
       'hyp_idf' (torch.Tensor): BxK, idf score of each word piece in the candidate sentence

    Returns
    A tensor that contains the F1 Bertscore for each example in the batch.
    """
    ref_embedding.div_(torch.norm(ref_embedding, dim=-1).unsqueeze(-1))
    hyp_embedding.div_(torch.norm(hyp_embedding, dim=-1).unsqueeze(-1))

    batch_size = ref_embedding.size(0)
    sim = torch.bmm(hyp_embedding, ref_embedding.transpose(1, 2))
    masks = torch.bmm(hyp_masks.unsqueeze(2).float(), ref_masks.unsqueeze(1).float())
    masks = masks.expand(batch_size, -1, -1).contiguous().view_as(sim)

    masks = masks.float().to(sim.device)
    sim = sim * masks

    word_precision = sim.max(dim=2)[0]
    word_recall = sim.max(dim=1)[0]

    hyp_idf.div_(hyp_idf.sum(dim=1, keepdim=True))
    ref_idf.div_(ref_idf.sum(dim=1, keepdim=True))
    precision_scale = hyp_idf.to(word_precision.device)
    recall_scale = ref_idf.to(word_recall.device)

    P = (word_precision * precision_scale).sum(dim=1)
    R = (word_recall * recall_scale).sum(dim=1)
    F = 2 * P * R / (P + R)

    F_bertscore = F.masked_fill(torch.isnan(F), 0.0)

    return F_bertscore