import torch
import torchvision.transforms.functional as TF
from torchmetrics import MeanMetric
from torchmetrics.functional import auroc

from data.utils.gaze_ops import get_multi_hot_map

from criteria.evals.metrics_utils import all_gather_on_cuda

class HeatmapAUC:
    def __init__(self, eval_name, heatmap_type = 'gaze', args=None):
        super().__init__()
        
        self.eval_name = eval_name
        self.gaze_heatmap_size = args.gaze_heatmap_size
        self.heatmap_type = heatmap_type
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

        if self.metric.device != outputs["pred_gaze_heatmap"].device:
            self.metric = self.metric.to(outputs["pred_gaze_heatmap"].device)
            
        tgt_gaze_points = torch.cat(
            [t["gaze_points"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )
        if 'gaze_points_is_padding' in targets[0].keys():
            tgt_gaze_points_is_padding = torch.cat(
                [t["gaze_points_is_padding"][i] for t, (_, i) in zip(targets, indices)], dim=0
            )
        else:
            tgt_gaze_points_is_padding = torch.full(tgt_gaze_points.shape[:2], False).to(tgt_gaze_points.device)

        tgt_watch_outsides = torch.cat(
            [t["gaze_watch_outside"][i] for t, (_, i) in zip(targets, indices)], dim=0
        ).bool()
        img_sizes = torch.cat(
            [t["img_size"][i] for t, (_, i) in zip(targets, indices)], dim=0
        ).reshape(-1, 2)

        pred_heatmaps = outputs["pred_gaze_heatmap"][idx].reshape(
            -1, self.gaze_heatmap_size, self.gaze_heatmap_size
        )


        for idx, (
            pred_heatmap,
            tgt_gaze_point,
            tgt_gaze_point_is_padding,
            tgt_watch_outside,
            img_size,
        ) in enumerate(
            zip(
                pred_heatmaps,
                tgt_gaze_points,
                tgt_gaze_points_is_padding,
                tgt_watch_outsides,
                img_sizes,
            )
        ):
            if tgt_watch_outside:
                continue

            img_height, img_width = img_size[0], img_size[1]
            pred_heatmap_scaled = torch.clip(pred_heatmap, 0, 1)
            pred_heatmap_scaled = TF.resize(
                pred_heatmap_scaled.unsqueeze(0),
                (img_height, img_width),
            ).squeeze()

            tgt_heatmap_scaled = get_multi_hot_map(
                tgt_gaze_point[~tgt_gaze_point_is_padding],
                (img_height, img_width),
                device=pred_heatmap_scaled.device,
            )

            auc_score = auroc(
                pred_heatmap_scaled.flatten(),
                tgt_heatmap_scaled.flatten(),
            )
            self.metric(auc_score.item())
