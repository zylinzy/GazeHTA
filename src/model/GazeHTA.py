import torch
import torch.nn as nn
import torch.nn.functional as F

from ldm.util import instantiate_from_config

from omegaconf import OmegaConf

from vpd.models import UNetWrapper


from timm.models.layers import trunc_normal_
from model.utils import build_conv_layer, build_norm_layer, build_upsample_layer

from mmengine.model import constant_init, normal_init
from lib.gaze_inout import GazeInOutLight


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class Encoder(nn.Module):
    def __init__(self, args, out_dim=1024, ldm_prior=[320, 640, 1280+1280], sd_path=None, text_dim=768,
                 conf_file='',
                 base_size =512,
                 ):
        super().__init__()

        # from 64 to 16
        self.layer1 = nn.Sequential(
            nn.Conv2d(ldm_prior[0], ldm_prior[0], 3, stride=2, padding=1),
            nn.GroupNorm(16, ldm_prior[0]),
            nn.ReLU(),
            nn.Conv2d(ldm_prior[0], ldm_prior[0], 3, stride=2, padding=1),
        )
        # from 32 to 16
        self.layer2 = nn.Sequential(
            nn.Conv2d(ldm_prior[1], ldm_prior[1], 3, stride=2, padding=1),
        )
        # from ch,ldm_prior to ch.out
        self.out_layer = nn.Sequential(
            nn.Conv2d(sum(ldm_prior), out_dim, 1),
            nn.GroupNorm(16, out_dim),
            nn.ReLU(),
        )
        
        self.apply(self._init_weights)
        
        ishead = False
        
        ### stable diffusion layers
        config = OmegaConf.load(conf_file)
        config.model.params.ckpt_path = f'{sd_path}'
        # remove text prompts
        config.model.params.cond_stage_config = '__is_unconditional__'
        
        sd_model = instantiate_from_config(config.model)
        self.encoder_vq = sd_model.first_stage_model
        
        self.unet = UNetWrapper(sd_model, use_attn=False, store_att = False)
    
        del sd_model.cond_stage_model
        del self.encoder_vq.decoder
        del self.unet.unet.diffusion_model.out

        for param in self.encoder_vq.parameters():
            param.requires_grad = False
            


    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, feats):
        x =  self.ldm_to_net[0](feats[0])
        for i in range(3):
            if i > 0:
                x = x + self.ldm_to_net[i](feats[i])
            x = self.layers[i](x)
            x = self.upsample_layers[i](x)
        return self.out_conv(x)

    def forward(self, latents):
        
        t = 1
        outs = self.unet(latents, t, c_crossattn=[None])
        

        feats = [outs[0], outs[1], torch.cat([outs[2], F.interpolate(outs[3], scale_factor=2)], dim=1)]
        x0 = torch.cat([self.layer1(feats[0]), self.layer2(feats[1]), feats[2]], dim=1)
        x = self.out_layer(x0)
        
        return x
        
    
class GazeHTA(nn.Module):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 args,
                 sd_path=None,
                 token_embed_dim=768,
                 neck_dim=[320, 640, 1280+1280],
                 conf_file='',
                 **kwargs):
        super().__init__()

        embed_dim = 192
        
        channels_in = embed_dim*8
        channels_out = embed_dim

        self.encoder = Encoder(args, out_dim=channels_in, ldm_prior=neck_dim, 
                               sd_path=sd_path,
                               text_dim=token_embed_dim,
                               conf_file=conf_file,
                               )
        
        self.encoder_vq = self.encoder.encoder_vq
        self.unet = self.encoder.unet
        
        self.decoder = Decoder(channels_in, channels_out, args.gaze_heatmap_size)
        
        if args.additional_connect != 0:
            out_ch = 3 * args.num_queries
        else:
            out_ch = 2 * args.num_queries
            
        if args.inject_heads_all != 0:
            added_ch = 1
            self.out_conv_1 = zero_module(nn.Conv2d(channels_out + added_ch, out_ch, kernel_size=3, stride=1, padding=1))
            self.last_layer_heatmaps_tmp = nn.Sequential(
                    nn.Conv2d(channels_out + added_ch, channels_out + added_ch, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=False))
        else:
            self.out_conv_1 = zero_module(nn.Conv2d(channels_out, out_ch, kernel_size=3, stride=1, padding=1))
            self.last_layer_heatmaps_tmp = nn.Sequential(
                    nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=False))
            
        self.last_layer_heatmaps = nn.Sequential(self.last_layer_heatmaps_tmp, self.out_conv_1)
                
        for m in self.last_layer_heatmaps_tmp.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
        
        
        if args.additional_head_heatmap_all != 0:
            self.out_conv_2 = zero_module(nn.Conv2d(channels_out, 1, kernel_size=3, stride=1, padding=1))
            self.last_layer_head_heatmap_all_tmp = nn.Sequential(
                nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=False))
            self.last_layer_head_heatmap_all = nn.Sequential(self.last_layer_head_heatmap_all_tmp, self.out_conv_2)
            
            for m in self.last_layer_head_heatmap_all_tmp.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001, bias=0)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
                    
        if args.additional_head_heatmap_all != 0 and args.inject_heads_all != 0:
            added_ch = 1
            self.inject_layer = zero_module(nn.Conv2d(added_ch, added_ch, kernel_size=1, stride=1))
            
        self.watch_outside_head = GazeInOutLight(channels_out, 64, args.num_queries, 1, ishead=False)
  
        self.num_queries = args.num_queries
        self.gaze_heatmap_size = args.gaze_heatmap_size
        self.additional_connect = args.additional_connect
        self.additional_head_heatmap_all = args.additional_head_heatmap_all
        self.inject_heads_all = args.inject_heads_all

    def forward(self, img):
        # image to latent torch.Size([B, 3, 576, 863])
        with torch.no_grad():
            latents = self.encoder_vq.encode(img).sample().detach() * 0.18215

        conv_feats = self.encoder(latents)

        # x: (B, 192, 64, 64)
        x = self.decoder([conv_feats])
        
        watch_outside = self.watch_outside_head(x).sigmoid()
  
        if self.inject_heads_all != 0:
            heatmap_all = self.last_layer_head_heatmap_all(x)
            heatmap_all_inject = self.inject_layer(heatmap_all)
            heatmaps = self.last_layer_heatmaps(torch.cat((x, heatmap_all_inject),dim=1))
            
            if self.additional_connect != 0:
                pred_gaze_heatmap = heatmaps[:, ::3, :, :]
                pred_head_heatmap = heatmaps[:, 1::3, :, :]
                pred_connect_heatmap = heatmaps[:, 2::3, :, :]
                output = {'pred_gaze_heatmap': pred_gaze_heatmap,
                            'pred_head_heatmap': pred_head_heatmap,
                            'pred_gaze_watch_outside': watch_outside,
                            'pred_connect_heatmap': pred_connect_heatmap}
            else:
                pred_gaze_heatmap = heatmaps[:, ::2, :, :]
                pred_head_heatmap = heatmaps[:, 1::2, :, :]
                
                output = {'pred_gaze_heatmap': pred_gaze_heatmap,
                            'pred_head_heatmap': pred_head_heatmap,
                            'pred_gaze_watch_outside': watch_outside}
            
            output['pred_head_heatmap_all'] = heatmap_all.squeeze(1)
        else:
            # heatmaps: (B, 192, 64, 64)
            heatmaps = self.last_layer_heatmaps(x)
            
            if self.additional_connect != 0:
                pred_gaze_heatmap = heatmaps[:, ::3, :, :]
                pred_head_heatmap = heatmaps[:, 1::3, :, :]
                pred_connect_heatmap = heatmaps[:, 2::3, :, :]
                output = {'pred_gaze_heatmap': pred_gaze_heatmap,
                            'pred_head_heatmap': pred_head_heatmap,
                            'pred_gaze_watch_outside': watch_outside,
                            'pred_connect_heatmap': pred_connect_heatmap}
            else:
                pred_gaze_heatmap = heatmaps[:, ::2, :, :]
                pred_head_heatmap = heatmaps[:, 1::2, :, :]
                
                output = {'pred_gaze_heatmap': pred_gaze_heatmap,
                            'pred_head_heatmap': pred_head_heatmap,
                            'pred_gaze_watch_outside': watch_outside}
            
        if self.additional_head_heatmap_all != 0 and self.inject_heads_all == 0:
            heatmap_all = self.last_layer_head_heatmap_all(x).squeeze(1)
            output['pred_head_heatmap_all'] = heatmap_all
            
        return output
    
import math
class Decoder(nn.Module): # input 16x16
    def __init__(self, in_channels, out_channels, out_size):
        super().__init__()
        
        self.in_channels = in_channels

        # import pdb; pdb.set_trace()
        scale_factor_log = int(math.log2(out_size) - math.log2(16))
        num_deconv = int(min(scale_factor_log, 3))
        self.num_up_sample = scale_factor_log - num_deconv
        #self.deconv = num_deconv
        num_filters = [32 for i in range(num_deconv)]
        num_kernels = [2 for i in range(num_deconv)]
        # 16 -> 32 -> 64 -> 128
        self.deconv_layers = self._make_deconv_layer(
            num_deconv,
            num_filters,
            num_kernels,
        )
        
        conv_layers = []
        conv_layers.append(
            build_conv_layer(
                dict(type='Conv2d'),
                in_channels=num_filters[-1],
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1))
        conv_layers.append(
            build_norm_layer(dict(type='BN'), out_channels)[1])
        conv_layers.append(nn.ReLU(inplace=True))
        self.conv_layers = nn.Sequential(*conv_layers)
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        self.init_weights()

    def forward(self, conv_feats):
        # import pdb; pdb.set_trace()
        # conv_feats[0]: B x ldm_dim x 16 x 16
        # out: B x 32 x 128 x 128 
        out = self.deconv_layers(conv_feats[0])
        # B x 192 x 128 x 128
        out = self.conv_layers(out)
        
        for i in range(self.num_up_sample):
            out = self.up(out)

        return out

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """Make deconv layers."""
        
        layers = []
        in_planes = self.in_channels
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                build_upsample_layer(
                    dict(type='deconv'),
                    in_channels=in_planes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            in_planes = planes

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel):
        """Get configurations for deconv layers."""
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError(f'Not supported num_kernels ({deconv_kernel}).')

        return deconv_kernel, padding, output_padding

    def init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
            elif isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)


   