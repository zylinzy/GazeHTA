import torch
import torch.nn as nn
import torch.nn.functional as F


class GazeWatchOutsideLoss(nn.Module):
    def __init__(self, loss_weight: int = 1, args = None):
        super().__init__()

        self.loss_weight = loss_weight
        self.args = args

    def forward(self, outputs, targets, indices, **kwargs):
        idx = kwargs["src_permutation_idx"]

        tgt_watch_outside = torch.cat(
            [t["gaze_watch_outside"][i] for t, (_, i) in zip(targets, indices)], dim=0
        ).flatten()

        pred_watch_outside = outputs["pred_gaze_watch_outside"][idx].flatten()

        loss = F.binary_cross_entropy_with_logits(
            pred_watch_outside, tgt_watch_outside.float()
        )

        return loss * self.loss_weight
