import torch
from torch import nn

from data.utils.box_ops import get_box_from_heatmap
from data.utils.gaze_ops import get_heatmap_peak_coords

class GOTDEvaluation(nn.Module):
    def __init__(
        self,
        matcher: nn.Module,
        evals: dict,
        args = None
    ):
        super().__init__()

        self.matcher = matcher
        self.evals = evals
        self.args = args

    @staticmethod
    def _get_src_permutation_idx(indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
    
    def calculate_scores(self, outputs):
        
        gaze_heatmaps = outputs['pred_gaze_heatmap']
        head_heatmaps = outputs['pred_head_heatmap']
        watch_outsides = outputs['pred_gaze_watch_outside']
        
        
        scores_sum_batch_all = []
        for i, (gaze_heatmap_batch, head_heatmap_batch, watch_outside_batch) in enumerate(zip(gaze_heatmaps, head_heatmaps, watch_outsides)):
            
            scores_sum_batch = []
            for j, (gaze_heatmap, head_heatmap, watch_outside) in enumerate(zip(gaze_heatmap_batch, head_heatmap_batch, watch_outside_batch)):
                
                gaze_heatmap_clipped = torch.clip(gaze_heatmap, 0, 1)
                head_heatmap_clipped = torch.clip(head_heatmap, 0, 1)
                
                best_box = get_box_from_heatmap(head_heatmap_clipped)
                x1, y1, x2, y2 = best_box[0].long()
                
                ###### [Confidence on head prediction] lowest_score if the max head response is lower than 0.1
                x1, y1, x2, y2 = best_box[0].long()
                conf_head = head_heatmap_clipped[y1:y2, x1:x2].max()
                
                x1, y1, x2, y2 = best_box[0].float()
                cx = (x1+x2)/2
                cy = (y1+y2)/2
                    
                ###### [Confidence on gaze prediction] get gaze center
                x, y = get_heatmap_peak_coords(gaze_heatmap_clipped)
                conf_gaze = gaze_heatmap_clipped[int(y), int(x)]
                
                ###### [Connection score] get gaze center
                if self.args.additional_connect != 0:
                     # draw gaze center
                    pred_gaze_point = torch.clip(torch.tensor([x, y]).round().int(), 0, self.args.gaze_heatmap_size - 1 )
                    pred_head_center= torch.clip(torch.tensor([cx, cy]).round().int(), 0, self.args.gaze_heatmap_size - 1 )
                    
                    # Connection scores
                    mid_num = 10
                    startend = list(zip(torch.linspace(pred_head_center[0], pred_gaze_point[0], steps=mid_num), \
                                    torch.linspace(pred_head_center[1], pred_gaze_point[1], steps=mid_num)))
                    
                    head_gaze_vector = pred_gaze_point.float() - pred_head_center.float()
                    norm = torch.norm(head_gaze_vector, p=2, dim=0)
                        
                    connect_heatmap_clipped = torch.clip(outputs['pred_connect_heatmap'][i, j], 0, 1)
                    score_midpts = torch.tensor([connect_heatmap_clipped[startend[I][1].round().int(), startend[I][0].round().int()] \
                                                    for I in range(len(startend))]).to(connect_heatmap_clipped.device)
                    score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(
                        0.5 * self.args.gaze_heatmap_size / norm - 1, 0)
                        
                else:
                    score_with_dist_prior = torch.tensor(0.0)     
                
                ###### [Final score]
                if watch_outside > 0.5:
                    scores_sum_batch.append((conf_head + (1-conf_gaze) - score_with_dist_prior)/3)
                else:
                    scores_sum_batch.append((conf_head + conf_gaze + score_with_dist_prior)/3)
                
            scores_sum_batch_all.append(torch.stack(scores_sum_batch))
            
        outputs['scores'] = torch.stack(scores_sum_batch_all, dim=0)    
       
        return outputs
    
    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the preds of the last layer and the targets
        outputs = self.calculate_scores(outputs)
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        
        indices = self.matcher(outputs_without_aux, targets, isEval = True)
        src_permutation_idx = self._get_src_permutation_idx(indices)
        self.indices_ = indices
        self.src_permutation_idx_ = src_permutation_idx
        
        for _, fn in self.evals.items():
            fn(
                outputs,
                targets,
                indices,
                src_permutation_idx=src_permutation_idx,
            )

    def reset(self):
        for _, fn in self.evals.items():
            fn.reset_metrics()

    def get_metrics(self):
        metrics = {}
        for _, fn in self.evals.items():
            metrics.update(fn.get_metrics())

        return metrics
