import torch
import torch.utils.data
from data.dataset import DatasetFactory
#from data.utils.NestedTensor import NestedTensor
from torch.utils.data.distributed import DistributedSampler

class CustomDatasetDataLoader:
    def __init__(self, args, mode):
        self.args = args
        self.mode = mode

        self.create_dataset()

    def create_dataset(self):

        self.dataset = DatasetFactory.get_by_name(self.args.dataset_mode, self.args, self.mode)

        # batch sampler
        if self.args.multi_gpu != 0 and self.mode == 'train':
            print(f"local rank {self.args.local_rank} / global rank {self.args.global_rank} successfully built train dataset.")
            data_sampler = DistributedSampler(self.dataset, shuffle=True, num_replicas=self.args.num_tasks, rank=self.args.global_rank)
            self.data_loader = torch.utils.data.DataLoader(
                            self.dataset, 
                            batch_size = self.args.batch_size,
                            sampler = data_sampler, 
                            shuffle = False,
                            num_workers = self.args.workers, 
                            pin_memory = self.args.pin_mem, 
                            drop_last = True,
                            collate_fn=self.collate_fn,)
        else:
            self.data_loader = torch.utils.data.DataLoader(
                            self.dataset, 
                            batch_size = self.args.batch_size,
                            shuffle = True if self.mode == 'train' else False,
                            pin_memory = self.args.pin_mem, 
                            drop_last = True,
                            collate_fn=self.collate_fn,)
        
    
    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

        This describes how to combine these tensors of different sizes. We use lists.

        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """
        #img, target, self.prompt_tokens #target["img_size"][0], target

        images = list()
        target = list()

        for b in batch:
            images.append(b[0])
            target.append(b[1])

        images = torch.stack(images, dim=0)

        return images, target  # tensor (N, 3, 300, 300), 3 lists of N tensors each
        
        
    def load_data(self):
        return self.data_loader

    def __len__(self):
        return len(self.dataset)
