import os


home_dir = '/path/to/the/working/folder/src/'

train_code = f'{home_dir}/train_gaze.py'


#### use VAT #######
epochs = 10
batch_size = 8
dataset_mode = 'data_videoAttTarget'
dataset = 'videoattentiontarget'
lr = 0.0004 # learning rate
down_fact_train = 0.2
####################

#### use GazeFollow #####
#epochs = 20
#batch_size = 8
#dataset_mode = 'data_gazefollow'
#dataset = 'gazefollow'
#lr = 0.000032 # learning rate
#down_fact_train = 1.0
#########################
    
data_dir = f'{home_dir}/train_data/{dataset}/'
img_size = 512
model_id = f'gaze_{dataset}'
gaze_heatmap_size = 64
num_queries = 20
output_dir = f'{home_dir}/results/{model_id}'
save_latest_freq_s = 14400 # save model every 'save_latest_freq_s' seconds
print_freq = 1 # log every 'print_freq' batch

#====================
# down-sampling factor for test/training set
down_fact_test = 1.0
#====================

# weights for each attribute for hungarian matching
matcher_head_weight = 1.0
matcher_gaze_weight = 2.5
matcher_inout_weight = 1.0
matcher_score_weight = 0.5 

head_heatmap_sigma = 5 # the sigma used to generate head heatmaps from the head bboxes

use_pseudo_head = 1 # whether to use labels from off-the-shelf head detector (here we use YOLO-v5)
no_validation = 0 # if '1', train model with all training samples, otherwise cut part of the training data out as validation set.

additional_connect = 1 # whether to prediction connection map
additional_head_heatmap_all = 1 # whether to predict head detection map (a heatmap for all the heads)
inject_heads_all = 1 # whether to re-inject the head features back

eval_only = 0 # whether to only run evaluation or not
use_pretrained = 0 # whether to initialize the model with a pretrained model
checkpoint_filename =  f'./pretrained_models/model_videoAttTarget_GazeHTA_epoch_9.pth' # path to the pre-trained model
#checkpoint_filename =  f'./pretrained_models/model_gazefollow_GazeHTA_epoch_19.pth' # path to the pre-trained model


cmd = f'python3.8 {train_code} \
                --pin_mem \
                --dataset_mode {dataset_mode} \
                --epochs {epochs} \
                --batch_size {batch_size}\
                --dataset {dataset}\
                --data_dir {data_dir}\
                --img_size {img_size}\
                --model_id {model_id}\
                --gaze_heatmap_size {gaze_heatmap_size}\
                --num_queries {num_queries}\
                --output_dir {output_dir}\
                --save_latest_freq_s {save_latest_freq_s}\
                --lr {lr} \
                --print_freq {print_freq}\
                --home_dir {home_dir}\
                --matcher_head_weight {matcher_head_weight}\
                --matcher_gaze_weight {matcher_gaze_weight}\
                --matcher_inout_weight {matcher_inout_weight}\
                --head_heatmap_sigma {head_heatmap_sigma}\
                --down_fact_test {down_fact_test}\
                --down_fact_train {down_fact_train}\
                --additional_connect {additional_connect}\
                --additional_head_heatmap_all {additional_head_heatmap_all}\
                --inject_heads_all {inject_heads_all}\
                --no_validation {no_validation}\
                --use_pseudo_head {use_pseudo_head}\
                --eval_only {eval_only}\
                --use_pretrained {use_pretrained}\
                --checkpoint_filename {checkpoint_filename}\
                --matcher_score_weight {matcher_score_weight}\
                '


os.system(cmd)
