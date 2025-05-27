import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='VPD training and testing')
    parser.add_argument('--amsgrad', action='store_true',
                        help='if true, set amsgrad to True in an Adam or AdamW optimizer.')
    parser.add_argument('-b', '--batch_size', default=8, type=int)
    parser.add_argument('--dataset', default='refcoco', help='refcoco, refcoco+, or refcocog')
    parser.add_argument('--device', default='cuda:0', help='device')  # only used when testing on a single machine
    parser.add_argument('--epochs', default=40, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--img_size', default=480, type=int, help='input image size')
    parser.add_argument("--local_rank", type=int, default=-1, help='local rank for DistributedDataParallel')
    parser.add_argument('--lr', default=0.00005, type=float, help='the initial learning rate')
    parser.add_argument('--model_id', default='vpd', help='name to identify the model')
    parser.add_argument('--output_dir', default='./checkpoints/', help='path where to save checkpoint weights')
    parser.add_argument('--pin_mem', action='store_true',
                        help='If true, pin memory when using the data loader.')
    parser.add_argument('--print_freq', default=10, type=int, help='print frequency (unit: # of batch)')
    parser.add_argument('--data_dir', default='', help='dataset root directory')
    parser.add_argument('--wd', '--weight-decay', default=1e-2, type=float, metavar='W', help='weight decay',
                        dest='weight_decay')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers')
    
    # ----------------------------
    parser.add_argument('--dataset_mode', default='data_videoAttTarget', help='dataset mode')
    parser.add_argument('--gaze_heatmap_size', default=64, type=int, help='output gaze heatmap size')
    parser.add_argument('--num_queries', default=20, type=int, help='number of heads per image')
    parser.add_argument('--save_latest_freq_s', default=14400, type=int, help='save model every XXX seconds (default: 4 hours)')
    parser.add_argument('--load_epoch', default=-1, type=int, help='the model to be resumed')
    parser.add_argument('--free_unet', default=1, type=int, help='whether to finetune diffusion unet')
    parser.add_argument('--home_dir', default='', help='working dir')
    
    # for testing/validate outside training
    parser.add_argument('--checkpoint_filename', default='', help='(test) specified model path')
    parser.add_argument('--run_mode', default='train', help='running in which mode', choices=['train', 'val', 'test'],)
    
    parser.add_argument('--matcher_head_weight', default=1, type=float, help='weight for head heatmap')
    parser.add_argument('--matcher_gaze_weight', default=1, type=float, help='weight for gaze heatmap')
    parser.add_argument('--matcher_inout_weight', default=1, type=float, help='weight for inout')
    
    
    parser.add_argument('--head_heatmap_sigma', default=3, type=float, help='the value used to derive the sigma for head heatmap')    
    parser.add_argument('--down_fact_test', default=0.2, type=float, help='down sample data factor')
    parser.add_argument('--down_fact_train', default=0.2, type=float, help='down sample data factor')    
    parser.add_argument('--gaze_heatmap_default_value', default=0.0, type=float, help='whether to predict the connection')
    
    parser.add_argument('--matcher_bbox_center_coeff', default=1, type=float, help='')
    
    parser.add_argument('--additional_connect', default=0, type=int, help='')
    parser.add_argument('--additional_head_heatmap_all', default=0, type=int, help='')
    parser.add_argument('--inject_heads_all', default=0, type=int, help='')
    
    parser.add_argument('--use_pretrained', default=0, type=int, help='')
    parser.add_argument('--eval_only', default=0, type=int, help='')
    
    parser.add_argument('--no_validation', default=0, type=int, help='')
    
    parser.add_argument('--multi_gpu', default=0, type=int, help='')
    
    parser.add_argument('--use_pseudo_head', default=0, type=int, help='')
    parser.add_argument('--matcher_score_weight', default=1, type=float, help='weight for score for matcher')
    
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args_dict = parser.parse_args()
