import torch
from torch import nn
from torch.nn import functional as F

from timm.models.layers import trunc_normal_
    
class GazeInOutLight(nn.Module):

    def __init__(self, in_ch, in_size, num_queries, num_layers, ishead=False):
        super().__init__()
        
        # downsample first
        self.in_size = in_size
        if in_size != 16:
            # from 64 to 16
            self.layer1 = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1),
                nn.GroupNorm(16, in_ch),
                nn.ReLU(),
                nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1),
            )
            
        
        out_dim = 16
        self.out_layer = nn.Sequential(
            nn.Conv2d(in_ch, out_dim, 1),
            nn.GroupNorm(16, out_dim),
            nn.ReLU(),
        )

        input_dim = out_dim * 16 * 16
        hidden_dim = input_dim
        output_dim = num_queries
        self.mlp_layers = MLP(input_dim, hidden_dim, output_dim, num_layers)

        self.ishead = ishead
        if ishead is True:
            self.mlp_layers_head = MLP(input_dim, hidden_dim, output_dim, num_layers)
            
        self.apply(self._init_weights)

    def forward(self, x):
        if self.in_size != 16:
            x = self.layer1(x)
        x = self.out_layer(x)
        # flatten to B, out_dim x 16 x 16
        x = x.flatten(start_dim=1)
        watch_outside = self.mlp_layers(x)

        if self.ishead:
            ishead = self.mlp_layers_head(x)

            return watch_outside, ishead
        else:
            return watch_outside
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)
    
class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x