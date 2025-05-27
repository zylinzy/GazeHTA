import numpy as np
import os
import time
import cv2
from tensorboardX import SummaryWriter


class TBVisualizer:
    def __init__(self, args):
        self._args = args
        self._save_path = os.path.join(self._args.output_dir)

        self._log_path = os.path.join(self._save_path, 'loss_log2.txt')
        self._tb_path = os.path.join(self._save_path, 'summary.json')
        self._writer = SummaryWriter(self._save_path)

        with open(self._log_path, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def __del__(self):
        self._writer.close()

    def plot_scalars(self, scalars, it, is_train):
        for label, scalar in scalars.items():
            sum_name = '{}/{}'.format('Train' if is_train else 'Val', label)
            self._writer.add_scalar(sum_name, scalar, it)
        self._writer.export_scalars_to_json(self._tb_path)

    def print_current_train_errors(self, epoch, i, iters_per_epoch, errors, t):
        log_time = time.strftime("[%d/%m/%Y %H:%M:%S]")
        message = '%s (T, epoch: %d, it: %d/%d, t/smpl: %.3fs) ' % (log_time, epoch, i, iters_per_epoch, t)
        for k, v in errors.items():
            message += '%s:%.5f ' % (k, v)

        print(message)
        with open(self._log_path, "a") as log_file:
            log_file.write('%s\n' % message)

    def print_current_validate_errors(self, epoch, errors, t):
        log_time = time.strftime("[%d/%m/%Y %H:%M:%S]")
        message = '%s (V, epoch: %d, time_to_val: %ds) ' % (log_time, epoch, t)
        for k, v in errors.items():
            message += '%s:%.5f ' % (k, v)

        print(message)
        with open(self._log_path, "a") as log_file:
            log_file.write('%s\n' % message)

