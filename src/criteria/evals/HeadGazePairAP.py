import torch
from torchmetrics import AveragePrecision
import torchvision.transforms.functional as TF

from data.utils.box_ops import box_iou, box_cxcywh_to_xyxy, get_box_from_heatmap
from data.utils.gaze_ops import get_heatmap_peak_coords, get_l2_dist

from criteria.evals.metrics_utils import all_gather_on_cuda

class HeadGazePairAP:
    def __init__(self, eval_name, args):
        super().__init__()
        
        self.eval_name = eval_name
        self.gaze_heatmap_size = args.gaze_heatmap_size
        self.metric = AveragePrecision() #task="binary")
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
        if self.metric.device != outputs["pred_head_heatmap"].device:
            self.metric = self.metric.to(outputs["pred_head_heatmap"].device)

        idx = kwargs["src_permutation_idx"]

        tgt_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0).unsqueeze(1)
        img_sizes = torch.cat([t["img_size"][i] for t, (_, i) in zip(targets, indices)], dim=0).reshape(-1, 2)

        pred_head_heatmaps = outputs["pred_head_heatmap"][idx].reshape(-1, self.gaze_heatmap_size, self.gaze_heatmap_size)
        
        tgt_gaze_points = torch.cat([t["gaze_points"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        if 'gaze_points_is_padding' in targets[0].keys():
            tgt_gaze_points_is_padding = torch.cat([t["gaze_points_is_padding"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        else:
            tgt_gaze_points_is_padding = torch.full(tgt_gaze_points.shape[:2], False).to(tgt_gaze_points.device)
              
        tgt_watch_outsides = torch.cat([t["gaze_watch_outside"][i] for t, (_, i) in zip(targets, indices)], dim=0).bool()
        
        pred_gaze_heatmaps = outputs["pred_gaze_heatmap"][idx].reshape(-1, self.gaze_heatmap_size, self.gaze_heatmap_size)
        
        for idx, (
                pred_head_heatmap, 
                tgt_box, 
                img_size, 
                pred_gaze_heatmap,
                tgt_gaze_point,
                tgt_gaze_point_is_padding,
                tgt_watch_outside,
            ) in enumerate(
                zip(
                    pred_head_heatmaps,
                    tgt_boxes,
                    img_sizes,
                    pred_gaze_heatmaps,
                    tgt_gaze_points,
                    tgt_gaze_points_is_padding,
                    tgt_watch_outsides,
                )
            ):  
            
            img_height, img_width = img_size[0], img_size[1]
            pred_head_heatmap_scaled = torch.clip(pred_head_heatmap, 0, 1)
            pred_head_heatmap_scaled = TF.resize(
                    pred_head_heatmap_scaled.unsqueeze(0),
                    (img_height, img_width),
                    ).squeeze()
            # cx, cy, w, h = tgt_box
            tgt_box_ = box_cxcywh_to_xyxy(tgt_box)
            pred_box = get_box_from_heatmap(pred_head_heatmap_scaled)
            # normalize box
            # x1, y1, x2, y2 = tgt_box
            tgt_box_[:, 0:1] = tgt_box_[:, 0:1] * img_width
            tgt_box_[:, 1:2] = tgt_box_[:, 1:2] * img_height
            tgt_box_[:, 2:3] = tgt_box_[:, 2:3] * img_width
            tgt_box_[:, 3:] = tgt_box_[:, 3:] * img_height

            iou, _ = box_iou(pred_box, tgt_box_)
            
            if tgt_watch_outside == False:
                pred_gaze_heatmap = torch.clip(pred_gaze_heatmap, 0, 1)
                pred_gaze_x, pred_gaze_y = get_heatmap_peak_coords(pred_gaze_heatmap)
                pred_gaze_coord_norm = (
                    torch.tensor(
                        [pred_gaze_x, pred_gaze_y], device=tgt_gaze_point_is_padding.device
                    )
                    / pred_gaze_heatmap.shape[0]  # NOTE: this assumes heatmap is square
                ).unsqueeze(0)

                # Average distance: distance between the predicted point and human average point
                mean_gt_gaze = torch.mean(
                    tgt_gaze_point[~tgt_gaze_point_is_padding], 0
                ).unsqueeze(0)
                
                avg_gaze_dist = get_l2_dist(mean_gt_gaze, pred_gaze_coord_norm)
                
                pred_matched_pair = ((iou > 0.5) & (avg_gaze_dist<0.15)).long()
            else:
                pred_matched_pair = ((iou > 0.5)).long()
            
            tgt_matched_pair = torch.full(pred_matched_pair.shape, 1).to(pred_matched_pair.device).long()
            self.metric(tgt_matched_pair, pred_matched_pair)
            