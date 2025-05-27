import torch
from torchmetrics import MeanMetric

from data.utils.gaze_ops import get_heatmap_peak_coords, get_l2_dist

from criteria.evals.metrics_utils import all_gather_on_cuda

class GazePointMinDistance:
    def __init__(self, eval_name, args):
        super().__init__()

        self.eval_name = eval_name
        self.gaze_heatmap_size = args.gaze_heatmap_size
        self.metric = MeanMetric()
        if args.multi_gpu != 0:
            self.metric.dist_sync_fn = all_gather_on_cuda 
            self.metric.dist_sync_on_step = False

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
        
        tgt_gaze_points = torch.cat([t["gaze_points"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        if 'gaze_points_is_padding' in targets[0].keys():
            tgt_gaze_points_is_padding = torch.cat([t["gaze_points_is_padding"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        else:
            tgt_gaze_points_is_padding = torch.full(tgt_gaze_points.shape[:2], False).to(tgt_gaze_points.device)
              
        tgt_watch_outsides = torch.cat([t["gaze_watch_outside"][i] for t, (_, i) in zip(targets, indices)], dim=0).bool()

        if self.metric.device != outputs["pred_gaze_heatmap"].device:
            self.metric = self.metric.to(outputs["pred_gaze_heatmap"].device)
            
        pred_heatmaps = outputs["pred_gaze_heatmap"][idx].reshape( -1, self.gaze_heatmap_size, self.gaze_heatmap_size)
        
        for idx, (
            pred_heatmap,
            tgt_gaze_point,
            tgt_gaze_point_is_padding,
            tgt_watch_outside,
        ) in enumerate(
            zip(
                pred_heatmaps,
                tgt_gaze_points,
                tgt_gaze_points_is_padding,
                tgt_watch_outsides,
            )
        ):
            if tgt_watch_outside:
                continue
            
            pred_heatmap = torch.clip(pred_heatmap, 0, 1)
            pred_gaze_x, pred_gaze_y = get_heatmap_peak_coords(pred_heatmap)
            pred_gaze_coord_norm = (
                torch.tensor(
                    [pred_gaze_x, pred_gaze_y], device=tgt_gaze_point_is_padding.device
                )
                / pred_heatmap.shape[0]  # NOTE: this assumes heatmap is square
            ).unsqueeze(0)

            all_distances = get_l2_dist(
                tgt_gaze_point[~tgt_gaze_point_is_padding], pred_gaze_coord_norm
            )

            self.metric(min(all_distances))
