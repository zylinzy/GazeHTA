import torch
import torch.nn as nn


class HeatmapLoss(nn.Module):
    def __init__(self, heatmap_type: str = 'gaze', loss_weight: int = 1, args = None):
        super().__init__()
        
        self.heatmap_type = heatmap_type
        self.loss_weight = loss_weight
        self.gaze_heatmap_size = args.gaze_heatmap_size
        self.args = args
        self.loss_fn = nn.MSELoss(reduction="none")

    def forward(self, outputs, targets, indices, **kwargs):
        idx = kwargs["src_permutation_idx"]
        
        if self.heatmap_type == 'head':
            tgt_heatmap = torch.cat(
                [t["head_heatmaps"][i] for t, (_, i) in zip(targets, indices)], dim=0
            )

            # If pred_gaze_heatmap is list, get the last one
            pred_heatmap = outputs["pred_head_heatmap"][idx]

        elif self.heatmap_type == 'connect_pair':
            tgt_heatmap = torch.cat(
                [t["connect_heatmaps"][i] for t, (_, i) in zip(targets, indices)], dim=0
            )
            
            pred_heatmap = outputs["pred_connect_heatmap"][idx] # (B x H ) x 64 x 64
            
        elif self.heatmap_type == 'head_all':
            tgt_heatmap = torch.stack([t["head_heatmaps_all"] for t in targets], dim=0) # (B, H, W)
            pred_heatmap = outputs["pred_head_heatmap_all"]
             
        else:
            tgt_heatmap = torch.cat(
                [t["gaze_heatmaps"][i] for t, (_, i) in zip(targets, indices)], dim=0
            )

            pred_heatmap = outputs["pred_gaze_heatmap"][idx]# (B x H ) x 64 x 64
            
        heatmap_loss = self.loss_fn(pred_heatmap, tgt_heatmap).mean()

        return heatmap_loss * self.loss_weight
