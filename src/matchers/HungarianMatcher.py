import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from data.utils.box_ops import get_box_from_heatmap
from data.utils.gaze_ops import get_heatmap_peak_coords

class HungarianMatcher(nn.Module):
    def __init__(
        self,
        args,
    ):
        super().__init__()

        self.args = args
        self.cost_gaze_watch_outside_coeff = args.matcher_inout_weight
        self.cost_gaze_heatmap_coeff = args.matcher_gaze_weight
        self.cost_head_heatmap_coeff = args.matcher_head_weight
            
        self.cost_bbox_center_coeff = args.matcher_bbox_center_coeff
            
        self.cost_score_coeff = args.matcher_score_weight
            
        assert (
            self.cost_gaze_watch_outside_coeff != 0,
            self.cost_gaze_heatmap_coeff != 0,
            self.cost_head_heatmap_coeff != 0
        ), "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets, isEval = False):
        bs, num_queries = outputs["pred_gaze_heatmap"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_gaze_heatmap = outputs["pred_gaze_heatmap"].flatten(
            0, 1
        ).flatten(1, 2)  # [batch_size * num_queries, gaze_heatmap_size]
        out_head_heatmap = outputs["pred_head_heatmap"].flatten(
            0, 1
        ).flatten(1, 2)
        out_watch_outside = outputs["pred_gaze_watch_outside"].flatten(
            0, 1
        ).unsqueeze(-1)  # [batch_size * num_queries, num_classes]
            
        if isEval == True: #[batch_size,  num_queries, 64, 64]
            out_bbox = []
            for head_heatmap_batch in outputs["pred_head_heatmap"]:
                for head_heatmap in head_heatmap_batch:
                    head_heatmap_ = torch.clip(head_heatmap, 0, 1)
                    best_box = get_box_from_heatmap(head_heatmap_)
                    x1, y1, x2, y2 = best_box[0].float()
                    cx = (x1+x2)/2
                    cy = (y1+y2)/2
                    out_bbox.append(torch.tensor([cx, cy]))
                    
            out_bbox = torch.stack(out_bbox, dim=0).to(outputs["pred_head_heatmap"].device)
            # normalize
            out_bbox = out_bbox / self.args.gaze_heatmap_size
            
            out_gaze_point = []
            for gaze_heatmap_batch in outputs["pred_gaze_heatmap"]:
                for gaze_heatmap in gaze_heatmap_batch:
                    # from heatmaps to bbox
                    gaze_heatmap_ = torch.clip(gaze_heatmap, 0, 1)
                    x, y = get_heatmap_peak_coords(gaze_heatmap_)
                    out_gaze_point.append(torch.tensor([x, y]))
            out_gaze_point = torch.stack(out_gaze_point, dim=0).to(outputs["pred_gaze_heatmap"].device)
            # normalize
            out_gaze_point = out_gaze_point / self.args.gaze_heatmap_size
            
        # Also concat the target labels and target
       
        tgt_gaze_heatmap = torch.cat([v["gaze_heatmaps"] for v in targets]).flatten(1, 2) # [batch_size * num_queries, gaze_heatmap_size]
        tgt_head_heatmap = torch.cat([v["head_heatmaps"] for v in targets]).flatten(1, 2)
        tgt_watch_outside = torch.cat([v["gaze_watch_outside"] for v in targets])
        
        if isEval == True:
            tgt_bbox = torch.cat([v["boxes"] for v in targets])
            tgt_bbox = tgt_bbox[:, :2]
            
            if self.args.dataset_mode == 'data_gazefollow':
                tgt_gaze_point = torch.cat([v["gaze_points"] for v in targets], dim=0)
                
                if 'gaze_points_is_padding' in targets[0].keys():
                    tgt_gaze_points_is_padding = torch.cat(
                        [v["gaze_points_is_padding"] for v in targets], dim=0)
                else:
                    tgt_gaze_points_is_padding = torch.full(tgt_gaze_point.shape[:2], False).to(tgt_gaze_point.device)
                    
                # loop over each sample in a batch
                tgt_gaze_point_out = []
                for tgt_gaze_point_b, tgt_gaze_points_is_padding_b in zip(tgt_gaze_point, tgt_gaze_points_is_padding):
                    tgt_gaze_point_out.append(tgt_gaze_point_b[~tgt_gaze_points_is_padding_b].mean(dim=0))
                tgt_gaze_point = torch.stack(tgt_gaze_point_out, dim=0)
            else:
                tgt_gaze_point = torch.cat([v["gaze_points"] for v in targets]).squeeze(1)
            
        #cost_class = -out_prob[:, tgt_ids.argmax(-1)]
        if isEval == True:
            cost_bbox_center = torch.cdist(out_bbox, tgt_bbox, p=2)
            cost_bbox_center[torch.isnan(cost_bbox_center)] = 0
                
            cost_gaze_heatmap1 = torch.cdist(out_gaze_point, tgt_gaze_point, p=2)
            cost_gaze_heatmap1[torch.isnan(cost_gaze_heatmap1)] = 0
                
            cost_gaze_heatmap1[:, tgt_watch_outside.squeeze(-1) == 0] = cost_gaze_heatmap1[:, tgt_watch_outside.squeeze(-1) == 0] * 0.5
            cost_gaze_heatmap1[:, tgt_watch_outside.squeeze(-1) == 1] = 0
            
            cost_gaze_heatmap2 = torch.cdist(out_gaze_heatmap, tgt_gaze_heatmap, p=2)
            cost_gaze_heatmap2[torch.isnan(cost_gaze_heatmap2)] = 0
            
            cost_gaze_heatmap = (cost_gaze_heatmap1 + cost_gaze_heatmap2 * 0.1 ) / 2
            
        else:
            cost_gaze_heatmap = torch.cdist(out_gaze_heatmap, tgt_gaze_heatmap, p=2)
            cost_gaze_heatmap[torch.isnan(cost_gaze_heatmap)] = 0
        
        cost_head_heatmap = torch.cdist(out_head_heatmap, tgt_head_heatmap, p=2)
        cost_head_heatmap[torch.isnan(cost_head_heatmap)] = 0
        
        cost_watch_outside = torch.abs(
            torch.cdist(out_watch_outside, tgt_watch_outside.float(), p=1)
        )
        cost_watch_outside[torch.isnan(cost_watch_outside)] = 0
        
        if isEval == True:
            cost_watch_outside[:, tgt_watch_outside.squeeze(-1) == 0] = cost_watch_outside[:, tgt_watch_outside.squeeze(-1) == 0] * 0.5
        
        if isEval == True:
            C = (
                self.cost_gaze_watch_outside_coeff * cost_watch_outside
                + self.cost_gaze_heatmap_coeff * cost_gaze_heatmap
                + self.cost_bbox_center_coeff * cost_bbox_center
            )
        else:
            C = (
                self.cost_gaze_watch_outside_coeff * cost_watch_outside
                + self.cost_gaze_heatmap_coeff * cost_gaze_heatmap
                + self.cost_head_heatmap_coeff * cost_head_heatmap
            )
            
        if isEval == True:
            pred_scores = outputs["scores"].flatten(
                0, 1
            ).unsqueeze(-1).float()
            
            tgt_scores = torch.full(tgt_watch_outside.shape, 1.0).to(pred_scores.device)
            
            cost_scores = torch.abs(
                torch.cdist(pred_scores, tgt_scores, p=1)
            )
            cost_scores[torch.isnan(cost_scores)] = 0
            
            C += self.cost_score_coeff * cost_scores
               
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [
            linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, dim=-1))
        ]
        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]
