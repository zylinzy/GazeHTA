import torch.utils.data as data
#from PIL import Image
import torchvision.transforms as transforms
import os

import data.utils.transforms as T


class DatasetFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_by_name(dataset_mode, args, mode):

        # INPUT = IMAGES:
        if dataset_mode == 'data_gazefollow':
            from data.data_gazefollow import Dataset
            dataset = Dataset(args, mode)
        elif dataset_mode == 'data_videoAttTarget':
            from data.data_videoAttTarget import Dataset
            dataset = Dataset(args, mode)
        else:
            raise ValueError("Dataset [%s] not recognized." % dataset_mode)

        print('Dataset {} was created'.format(dataset.name))
        return dataset

class DatasetBase(data.Dataset):
    def __init__(self, opt, mode):
        super(DatasetBase, self).__init__()
        self._name = 'BaseDataset'
        self._root = None
        self._opt = opt
        self._mode = mode
        self._create_transform()

        self._IMG_EXTENSIONS = [
            '.jpg', '.JPG', '.jpeg', '.JPEG',
            '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
        ]

    @property
    def name(self):
        return self._name

    @property
    def path(self):
        return self._root

    def _create_transform(self):
        normalize = T.Compose(
                [T.ToTensor(), T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
            )

        if self._mode == 'train':
            img_transform = T.Compose(
                [
                    T.RandomHorizontalFlip(),
                    T.RandomCrop(p=0.5),
                    T.RandomResize([self._opt.img_size], max_size=1333),
                    normalize,
                ]
            )
        else:
            img_transform = T.Compose(
                [
                    T.RandomResize([self._opt.img_size], max_size=1333),
                    normalize,
                ]
            )

        self._transform = img_transform

    def get_transform(self):
        return self._transform

    def _is_image_file(self, filename):
        return any(filename.endswith(extension) for extension in self._IMG_EXTENSIONS)

    def _is_csv_file(self, filename):
        return filename.endswith('.csv')

    def _get_all_files_in_subfolders(self, dir, is_file):
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir

        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

        return images
