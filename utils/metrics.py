from typing import Any, Optional, List

from pytorch_lightning.metrics import SklearnMetric

try:
    from torch.distributed import ReduceOp, group
except ImportError:
    class ReduceOp:
        SUM = None


    class group:
        WORLD = None

import numpy as np


class ClassificationReport(SklearnMetric):
    """
    Calculates the Classification Report

    Warning:
            Every metric call will cause a GPU synchronization, which may slow down your code
    """

    def __init__(
            self,
            reduce_group: Any = group.WORLD,
            reduce_op: Any = ReduceOp.SUM,
    ):
        """
        Args:
            reduce_group: the process group for DDP reduces (only needed for DDP training).
                Defaults to all processes (world)
            reduce_op: the operation to perform during reduction within DDP (only needed for DDP training).
                Defaults to sum.
        """
        super().__init__(metric_name='classification_report',
                         reduce_group=reduce_group,
                         reduce_op=reduce_op)

    def forward(
            self,
            y_pred: np.ndarray,
            y_true: np.ndarray,
            target_names: Optional[List[str]] = None,
            sample_weight: Optional[np.ndarray] = None
    ) -> float:
        """
        Computes the accuracy

        Args:
            y_pred: the array containing the predictions (already in categorical form)
            y_true: the array containing the targets (in categorical form)
            target_names: display names matching the labels (same order)
            sample_weight:  sample weights

        Return:
            Accuracy Score
        """
        return super().forward(y_pred=y_pred, y_true=y_true, target_names=target_names, sample_weight=sample_weight)


# https://github.com/bearpaw/pytorch-classification/blob/cc9106d598ff1fe375cc030873ceacfea0499d77/utils/eval.py
def top_k_accuracy(output, target, topk=(1,)):
    """Computes the accuracy @k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = dict()
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res[f'top_{k}_acc'] = correct_k.mul_(100.0 / batch_size)
    return res
