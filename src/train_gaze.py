import datetime
import os
import time

import torch
import torch.utils.data
import warnings
warnings.filterwarnings("ignore")

import utils
import numpy as np

import gc
from collections import OrderedDict

from data.custom_dataset_data_loader import CustomDatasetDataLoader
from criteria.GOTDSetCriterion import GOTDSetCriterion
from criteria.GOTDEvaluation import GOTDEvaluation
from criteria.losses.HeatmapLoss import HeatmapLoss

from criteria.losses.GazeWatchOutsideLoss import GazeWatchOutsideLoss
from criteria.evals.HeatmapAUC import HeatmapAUC
from criteria.evals.GazePointAvgDistance import GazePointAvgDistance
from criteria.evals.GazePointMinDistance import GazePointMinDistance
from criteria.evals.GazeWatchOutsideAP import GazeWatchOutsideAP
from criteria.evals.HeadGazePairAP import HeadGazePairAP

from matchers.HungarianMatcher import HungarianMatcher
from tb_visualizer import TBVisualizer

import tqdm

from args import get_parser
import utils
import copy

from utils import revert_sync_batchnorm
import random

class Evaluate:
    def __init__(self, args, model, criterion, criterion_eval, device):
        
        args.run_mode = 'test'
        self.args = args
        self.args.batch_size = 1
        self.model = model
        self.device = device
        self.run_mode = args.run_mode 
        self.args.down_fact_test = 1.0
        # --------------------------------
        # data loader
        # --------------------------------
        _data_loader_test = CustomDatasetDataLoader(args, self.run_mode)
        self.data_loader_test = _data_loader_test.load_data()
        
        self.criterion = copy.deepcopy(criterion)
        self.criterion_per_sample = copy.deepcopy(criterion)
        
        # ------------------
        # set criteria (eval)
        # ------------------
        criterion_eval.reset()
        self.criterion_eval = copy.deepcopy(criterion_eval)
        self.criterion_eval_per_sample = copy.deepcopy(criterion_eval)
        
        # --------------------------------
        # tools for tensorboard
        # -------------------------------- 
        self.log_path = os.path.join(args.output_dir, f'test_log.txt')
        os.makedirs(args.output_dir, exist_ok=True)

        self.log_path_per_img = os.path.join(args.output_dir, f'test_log_per_img.txt')
        os.makedirs(args.output_dir, exist_ok=True)

        with open(self.log_path, "w") as log_file:
            now = time.strftime("%c")
            log_file.write('================ (%s) ================\n' % now)
        with open(self.log_path_per_img, "w") as log_file:
            now = time.strftime("%c")
            log_file.write('================ (%s) ================\n' % now)
            
        self.evaluate()

    def log_results(self, epoch, errors, t, log_path, prefix = ''):
        log_time = time.strftime("[%d/%m/%Y %H:%M:%S]")
        message = '%s (%s, i_batch: %d, %s, time_to_val: %ds) ' % (log_time, 'Val' if self.run_mode == 'val' else 'Test', epoch, prefix, t)
        for k, v in errors.items():
            message += '%s:%.5f ' % (k, v)

        print(message)
        with open(log_path, "a") as log_file:
            log_file.write('%s\n' % message)

    def evaluate(self):
        self.model.eval()
        if utils.is_dist_avail_and_initialized():
            self.model = revert_sync_batchnorm(self.model)
        total_its = 0
        self.criterion_eval.reset()

        val_errors = OrderedDict()
        val_start_time = time.time()
        
        with torch.no_grad():
            
            for i_batch, data in enumerate(tqdm.tqdm(self.data_loader_test)):
                total_its += 1
                image, target = data
                image = image.cuda(non_blocking=True)
                
                for i_h in range(len(target)):
                    for key, _ in target[i_h].items():
                        if torch.is_tensor(target[i_h][key]):
                            target[i_h][key] = target[i_h][key].cuda(non_blocking=True)

                # ---------------------------------
                # prepare token and embeddings
                # ---------------------------------
                output = self.model(image)

                for key, _ in output.items():
                    output[key] = output[key].cpu()#.data.numpy()

                for i_h in range(len(target)):
                    for key, _ in target[i_h].items():
                        if torch.is_tensor(target[i_h][key]):
                            target[i_h][key] = target[i_h][key].cpu()#.data.numpy()

                losses = self.criterion(output, target)
                self.criterion_eval(output, target)

                # store current batch errors
                for k, v in losses.items():
                    if k in val_errors:
                        val_errors[k] += v.cpu().data.numpy()
                    else:
                        val_errors[k] = v.cpu().data.numpy()

                if 'total_loss' in val_errors:
                    val_errors['total_loss'] += sum(losses.values()).cpu().data.numpy()
                else:
                    val_errors['total_loss'] = sum(losses.values()).cpu().data.numpy()

                # per sample result
                losses_ = self.criterion_per_sample(output, target)
                self.criterion_eval_per_sample.reset()
                self.criterion_eval_per_sample(output, target)
                results_ = self.criterion_eval_per_sample.get_metrics()
                prefix = target[0]['path']
                for k, v in losses_.items():
                    if k in results_:
                        results_[k] += v.cpu().data.numpy()
                    else:
                        results_[k] = v.cpu().data.numpy()
                if self.args.global_rank == 0:
                    self.log_results(i_batch, results_, time.time() - val_start_time, self.log_path_per_img, prefix)

                        
                del image, target, losses, output, data

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    
                #if i_batch == 4:
                #    break
            # normalize errors
            for k in val_errors.keys():
                val_errors[k] /= total_its

            results = self.criterion_eval.get_metrics()
            for k, v in results.items():
                val_errors[k] = v

            # display and plot curve
            if self.args.global_rank == 0:
                self.log_results(100000, val_errors, time.time() - val_start_time, self.log_path)
            
        self.model.train()
        if utils.is_dist_avail_and_initialized():
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

class Train:
    def __init__(self):
        
        parser = get_parser()
        args = parser.parse_args()

        # set up distributed learning
        if args.multi_gpu != 0:
            utils.init_distributed_mode(args)
        else:
            args.local_rank = 0
            args.world_size = 1
            
        print('utils.is_dist_avail_and_initialized()', utils.is_dist_avail_and_initialized())
        print(f"local rank {args.local_rank} / global rank {utils.get_rank()} successfully built train dataset.")
        args.num_tasks = utils.get_world_size()
        args.global_rank = utils.get_rank()

        utils.print_args(args)
        utils.set_and_check_load_epoch(args)
        
        self.args = args

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # --------------------------------
        # data loader
        # --------------------------------
        _data_loader_train = CustomDatasetDataLoader(args, 'train')
        self.args.down_fact_test = 0.2
        if args.no_validation != 0:
            _data_loader_val = CustomDatasetDataLoader(args, 'test')
        else:    
            _data_loader_val = CustomDatasetDataLoader(args, 'val')

        self.data_loader_train = _data_loader_train.load_data()
        self.data_loader_val = _data_loader_val.load_data()

        # --------------------------------
        # Prepare model
        # -------------------------------- 
        from model.GazeHTA import GazeHTA
        self.model = GazeHTA( args, 
                    sd_path=f'{args.home_dir}/checkpoints/v1-5-pruned-emaonly.ckpt',
                    conf_file=f'{args.home_dir}/v1-inference.yaml',
                    neck_dim = [320, 640, 1280+1280]
                    )
        
        if utils.is_dist_avail_and_initialized():
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model.cuda()
        
        # parameters to optimize
        lesslr_no_decay = list()
        lesslr_decay = list()
        no_lr = list()
        no_decay = list()
        decay = list()
        if args.free_unet == 1:
            no_grad_list = ['encoder_vq']
        else:
            no_grad_list = ['unet.unet', 'encoder_vq']

        for name, m in self.model.named_parameters():
            # disable gradients on unet.unet
            is_continue = False
            for sub in no_grad_list:
                if sub in name:
                    is_continue = True
            if is_continue:
                continue

            if 'unet' in name and 'norm' in name:
                lesslr_no_decay.append(m)
            elif 'unet' in name:
                lesslr_decay.append(m)
            elif 'encoder_vq' in name:
                no_lr.append(m)
            elif 'norm' in name:
                no_decay.append(m)
            else:
                decay.append(m)
        
        params_to_optimize = [
            {'params': lesslr_no_decay, 'weight_decay': 0.0, 'lr_scale':0.01},
            {'params': lesslr_decay, 'lr_scale': 0.01},
            {'params': no_lr, 'lr_scale': 0.0},
            {'params': no_decay, 'weight_decay': 0.0},
            {'params': decay}
        ]

        
        # optimizer
        self.optimizer = torch.optim.AdamW(params_to_optimize,
                                    lr=args.lr,
                                    weight_decay=args.weight_decay,
                                    amsgrad=args.amsgrad
                                    )
        
                                    
        # Summary
        if self.args.global_rank == 0:
            print('---------- Data Summary ---------------')
            print('#train images = %d' % len(self.data_loader_train.dataset))
            print('#val images = %d' % len(self.data_loader_val.dataset))
            print('---------- Model Summary ---------------')
            unet_num = sum(p.numel() for p in self.model.encoder.unet.unet.parameters() if p.requires_grad)
            if 'unet.unet' not in no_grad_list:
                print('---unet.unet parameters : {}'.format(unet_num))
            print('---encoder parameters : {}'.format(sum(p.numel() for p in self.model.encoder.parameters() if p.requires_grad)-unet_num))
            print('---decoder parameters : {}'.format(sum(p.numel() for p in self.model.decoder.parameters() if p.requires_grad)))
        
            print('---last_layer_heatmaps parameters : {}'.format(sum(p.numel() for p in self.model.last_layer_heatmaps.parameters() if p.requires_grad)))
            print('----------------------------------------')

        # learning rate scheduler
        total_steps = len(self.data_loader_train) * args.epochs
        self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,max_lr=args.lr, total_steps=total_steps)
        

        # resume training (optimizer, lr scheduler, and the epoch)
        if args.load_epoch >= 0:
            checkpoint_path = os.path.join(args.output_dir,'model_{}_epoch_{}.pth'.format(args.model_id, args.load_epoch))
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['model'])
            if args.load_epoch < args.epochs - 1:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                self.resume_epoch = checkpoint['epoch']
            else:
                self.resume_epoch = args.load_epoch
            del checkpoint  # dereference seems crucial
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.tot_seen_data = (self.resume_epoch +1) * len(self.data_loader_train) * self.args.batch_size * self.args.world_size
            self.tot_steps = (self.resume_epoch +1) * len(self.data_loader_train)
            if self.args.global_rank == 0:
                print(f'Reload from checkpoint {checkpoint_path}')
                print('----------------------------------------')
        else:
            self.resume_epoch = -999
            self.tot_seen_data = 0
            self.tot_steps = 0
        
        if args.use_pretrained != 0:
            checkpoint = torch.load(args.checkpoint_filename, map_location='cpu')
            self.model.load_state_dict(checkpoint['model'])
            del checkpoint  # dereference seems crucial
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        if utils.is_dist_avail_and_initialized():   
            model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.args.local_rank], find_unused_parameters=True)
            self.model = model.module
        
        # ------------------
        # set criteria
        # ------------------
        loss_criterion = {}
        loss_head_heatmap = HeatmapLoss(heatmap_type = 'head', loss_weight = args.matcher_head_weight, args = args)
        loss_gaze_heatmap = HeatmapLoss(heatmap_type = 'gaze', loss_weight = args.matcher_gaze_weight, args = args)
        loss_gaze_outside = GazeWatchOutsideLoss(loss_weight = args.matcher_inout_weight, args = args)
        matcher = HungarianMatcher(args)
        
        loss_criterion = {'loss_head': loss_head_heatmap, 
                          'loss_gaze': loss_gaze_heatmap,
                          'loss_outside': loss_gaze_outside}
        
        if args.additional_connect != 0:
            loss_connect_heatmap = HeatmapLoss(heatmap_type = 'connect_pair', loss_weight = args.matcher_gaze_weight, args = args)
            loss_criterion['loss_conected'] = loss_connect_heatmap
          
        if args.additional_head_heatmap_all != 0:
            loss_head_heatmap_all = HeatmapLoss(heatmap_type = 'head_all', loss_weight = args.matcher_head_weight, args = args)
            loss_criterion['loss_head_all'] = loss_head_heatmap_all
            
        self.criterion = GOTDSetCriterion(matcher, losses = loss_criterion, args=args)
        
        # ------------------
        # set criteria (eval)
        # ------------------
        eval_gaze_avg_dist = GazePointAvgDistance('eval_gaze_avg_dist', args = args)
        eval_gaze_auc = HeatmapAUC('eval_gaze_auc', heatmap_type = 'gaze', args = args)
        eval_head_gaze_map = HeadGazePairAP('eval_head_gaze_map', args = args)
        
        evals =  {'eval_gaze_avg_dist': eval_gaze_avg_dist,
                    'eval_gaze_auc': eval_gaze_auc,
                    'eval_head_gaze_map': eval_head_gaze_map}
        
        if args.dataset_mode == 'data_gazefollow':
            eval_gaze_min_dist = GazePointMinDistance('eval_gaze_min_dist', args = args)
            evals['eval_gaze_min_dist'] = eval_gaze_min_dist
        elif args.dataset_mode == 'data_videoAttTarget':
            eval_gaze_outside = GazeWatchOutsideAP('eval_gaze_outside', args = args)
            evals['eval_gaze_outside'] = eval_gaze_outside
            
        self.criterion_eval = GOTDEvaluation(matcher, evals = evals, args=args)
        

        # --------------------------------
        # tools for tensorboard
        # -------------------------------- 
        if self.args.global_rank == 0:
            self.tb_visualizer = TBVisualizer(args)
        self.last_save_latest_time = None
        # ----
        self.device = device
        
        self.train()

    def validate(self):
        
        self.model.eval()
        total_its = 0
        if utils.is_dist_avail_and_initialized():
            self.model = revert_sync_batchnorm(self.model)

        self.criterion_eval.reset()

        val_errors = OrderedDict()
        val_start_time = time.time()
        
        with torch.no_grad():
            
            for i_batch, data in enumerate(self.data_loader_val):
                total_its += 1
                image, target = data

                image = image.cuda(non_blocking=True)
                
                for i_h in range(len(target)):
                    for key, _ in target[i_h].items():
                        if torch.is_tensor(target[i_h][key]):
                            target[i_h][key] = target[i_h][key].cuda(non_blocking=True)
 
                # ---------------------------------
                # prepare token and embeddings
                # ---------------------------------
                # run model
                output = self.model(image)
                

                for key, _ in output.items():
                    output[key] = output[key].cpu()#.data.numpy()

                for i_h in range(len(target)):
                    for key, _ in target[i_h].items():
                        if torch.is_tensor(target[i_h][key]):
                            target[i_h][key] = target[i_h][key].cpu()#.data.numpy()

                losses = self.criterion(output, target)
                self.criterion_eval(output, target)

                # store current batch errors
                for k, v in losses.items():
                    if k in val_errors:
                        val_errors[k] += v.cpu().data.numpy()
                    else:
                        val_errors[k] = v.cpu().data.numpy()

                if 'total_loss' in val_errors:
                    val_errors['total_loss'] += sum(losses.values()).cpu().data.numpy()
                else:
                    val_errors['total_loss'] = sum(losses.values()).cpu().data.numpy()
                        
                del image, target, losses, output, data

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    
                #if i_batch == 4:
                #    break


            # normalize errors
            for k in val_errors.keys():
                val_errors[k] /= total_its

            results = self.criterion_eval.get_metrics()
            for k, v in results.items():
                val_errors[k] = v

            # display and plot curve
            t = (time.time() - val_start_time)
            if self.args.global_rank == 0:
                self.tb_visualizer.print_current_validate_errors(self.i_epoch, val_errors, t)
                self.tb_visualizer.plot_scalars(val_errors, self.tot_steps, is_train=False)
                
            
        # set back to train
        self.model.train()
        if utils.is_dist_avail_and_initialized():
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)


    def train_one_epoch(self):
        
        self.model.train()
        print_freq = self.args.print_freq
        iters_per_epoch = len(self.data_loader_train)
        total_its = 0

        for i_batch, data in enumerate(self.data_loader_train):
            iter_start_time = time.time()
            total_its += 1
            image, target = data
            image = image.cuda(non_blocking=True)
            
            for i_h in range(len(target)):
                for key, _ in target[i_h].items():
                    if torch.is_tensor(target[i_h][key]):
                        target[i_h][key] = target[i_h][key].cuda(non_blocking=True)

            # ---------------------------------
            # prepare token and embeddings
            # ---------------------------------
            output = self.model(image)

            losses = self.criterion(output, target)
            loss = sum(losses.values())
                
            self.optimizer.zero_grad(set_to_none=True)  # set_to_none=True is only available in pytorch 1.6+
            loss.backward()
            
            self.optimizer.step()
            self.lr_scheduler.step()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            self.tot_seen_data += self.args.batch_size * self.args.world_size
            self.tot_steps += 1
            

            # display (on terminal) training results every 'print_freq' step
            if i_batch == 0  or (i_batch % print_freq == 0) or (i_batch == iters_per_epoch-1):
                errors = {}
                # store current batch errors
                for k, v in losses.items():
                    errors[k] = v.cpu().data.numpy()

                errors['total_loss'] = loss.cpu().data.numpy()
                t = (time.time() - iter_start_time) / (self.args.batch_size * self.args.world_size)
                if self.args.global_rank == 0:
                    self.tb_visualizer.print_current_train_errors(self.i_epoch, i_batch, iters_per_epoch, errors, t)
                    self.tb_visualizer.plot_scalars(errors, self.tot_steps, is_train=True)
                    lr_scalar = {'lr': self.optimizer.param_groups[0]["lr"]}
                    self.tb_visualizer.plot_scalars(lr_scalar, self.tot_steps, is_train=True)


            # save model every 'save_latest_freq_s' seconds
            if self.last_save_latest_time is None or time.time() - self.last_save_latest_time > self.args.save_latest_freq_s:
                if self.args.global_rank == 0:
                    print('saving the latest model (epoch %d, total_steps %d)' % (self.i_epoch, self.tot_steps))
                dict_to_save = {'model': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(), 'epoch': self.i_epoch,
                            'lr_scheduler': self.lr_scheduler.state_dict()}
                
                utils.save_on_master(dict_to_save, os.path.join(self.args.output_dir,
                                    'model_{}_epoch_{}.pth'.format(self.args.model_id, self.i_epoch)))
                self.last_save_latest_time = time.time()

            del image, target, losses, output, data

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
            #if i_batch == 4:
            #    break
                


    def train(self):
        # --------------------------------
        # training loops
        # -------------------------------- 
        # housekeeping
        if self.args.eval_only != 0:
            start_time = time.time()
            if self.args.global_rank == 0:
                print('===================================')
                print('Testing...')
                Evaluate(self.args, self.model, self.criterion, self.criterion_eval, self.device)
                
                # summarize
                total_time = time.time() - start_time
                total_time_str = str(datetime.timedelta(seconds=int(total_time)))
                print('Testing time {}'.format(total_time_str))
        else:
            start_time = time.time()
            self.i_epoch = self.resume_epoch
            for epoch in range(max(0, self.resume_epoch+1), self.args.epochs):
                if utils.is_dist_avail_and_initialized():
                    self.data_loader_train.sampler.set_epoch(epoch)
                
                self.i_epoch = epoch
                
                self.train_one_epoch()
                # ------------------
                # Validate gaze target detection after one epoch
                # multi-person: https://github.com/francescotonini/object-aware-gaze-target-detection
                # ------------------
                self.validate()
                if self.args.global_rank == 0:
                    print('saving the model at the end of epoch %d, iters %d' % (epoch, self.tot_steps))
                dict_to_save = {'model': self.model.state_dict(),
                                'optimizer': self.optimizer.state_dict(), 'epoch': epoch,
                                'lr_scheduler': self.lr_scheduler.state_dict()}

                utils.save_on_master(dict_to_save, os.path.join(self.args.output_dir,
                                    'model_{}_epoch_{}.pth'.format(self.args.model_id, epoch)))
                
                ### remove old 
                if epoch > 0:
                    pre_file = os.path.join(self.args.output_dir,
                                        'model_{}_epoch_{}.pth'.format(self.args.model_id, epoch-1))
                    if os.path.exists(pre_file):
                        os.system(f'rm -rf {pre_file}')
                
            # do testing in the last epoch
            if self.args.global_rank == 0:
                print('===================================')
                print('Validation at the end of epoch %d, iters %d' % (self.args.epochs-1, self.tot_steps))
            self.validate()
            if self.args.global_rank == 0:
                print('===================================')
                print('Testing at the end of epoch %d, iters %d' % (self.args.epochs-1, self.tot_steps))
            Evaluate(self.args, self.model, self.criterion, self.criterion_eval, self.device)
            
            # summarize
            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            if self.args.global_rank == 0:
                print('Training time {}'.format(total_time_str))

def set_random_seeds(random_seed):
    os.environ['PYTHONHASHSEED']=str(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True, warn_only=True)
    
if __name__ == "__main__":
    
    set_random_seeds(1234)
    
    Train()

