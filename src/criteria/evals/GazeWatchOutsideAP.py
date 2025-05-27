import torch
from torchmetrics import AveragePrecision

from criteria.evals.metrics_utils import all_gather_on_cuda

class GazeWatchOutsideAP:
    def __init__(self, eval_name, args):
        super().__init__()

        self.metric = AveragePrecision()
        if args.multi_gpu != 0:
            self.metric.dist_sync_fn = all_gather_on_cuda 
            self.metric.dist_sync_on_step = False
        self.eval_name = eval_name

    def reset_metrics(self):
        self.metric.reset()

    def get_metrics(self):
        return {
            self.eval_name: self.metric.compute().item(),
        }

    @torch.no_grad()
    def __call__(self, outputs, targets, indices, **kwargs):
        # If metric is not on the same device as outputs, put it
        # on the same device as outputs
        idx = kwargs["src_permutation_idx"]

        if self.metric.device != outputs["pred_gaze_watch_outside"].device:
            self.metric = self.metric.to(outputs["pred_gaze_watch_outside"].device)

        pred_watch_outside = outputs["pred_gaze_watch_outside"][idx].flatten()
            
        tgt_watch_outside = (
            torch.cat(
                [t["gaze_watch_outside"][i] for t, (_, i) in zip(targets, indices)],
                dim=0,
            ).flatten()
            .long()
        )

        # If there are no targets, return
        if len(tgt_watch_outside) == 0 or len(pred_watch_outside) == 0:
            return

        self.metric(pred_watch_outside, tgt_watch_outside)
