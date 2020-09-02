# https://github.com/fastai/fastai/blob/bdb3e420706c73516d0a72f802bb1aa9e7e303a7/fastai/metrics.py#L109

def top_k_accuracy(y_pred, y_true, k_values, axis=-1):
    """Computes the Top-k accuracy (`targ` is in the top `k` predictions of `inp`)"""
    res = dict()
    for k in k_values:
        inp = y_pred.topk(k=k, dim=axis)[1]
        out = y_true.unsqueeze(dim=axis).expand_as(inp)
        res[f'top_{k}_acc'] = (inp == out).sum(dim=-1).float().mean()
    return res
