import os
from copy import deepcopy
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data.dataset import Subset

from datasets.augs import augmentations
from datasets.cifar_mod import CIFAR100, CIFAR10
from datasets.mnist3d import MNIST3D as MNIST
from datasets.svhn_mod import SVHN
from datasets.class_split import ClassSplit
from datasets.custom_imagefolder import ImageFolder

class StandardCL:
    """Create CL problem. For simplicity, assume equal number of classes per task"""
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.args = args

        self.validation = args.validation
        
        self.n_cls_task = args.n_cls // args.n_tasks
        self.cls_list = np.arange(args.n_cls).reshape(args.n_tasks, -1).tolist()

    def make_dataset(self, task_id):
        """Make a CL dataset for task = task_id"""
        dataset_ = deepcopy(self.dataset)

        if self.validation is not None:
            dataset_valid = deepcopy(self.dataset)

        idx_list, idx_list_valid = [], [] # These are used for ImageNet
        
        for c in self.cls_list[task_id]:
            idx = np.where(self.dataset.targets == c)[0]
            
            if self.args.exe:
                idx = idx[:self.args.exe_n_samples]
            
            if self.validation is not None:
                if self.args.seed != 0:
                    np.random.shuffle(idx)
                n_samples = len(idx)
                idx_valid = idx[int(n_samples * self.validation):]
                idx = idx[:int(n_samples * self.validation)]
            
            idx_list.append(idx)
            if self.validation is not None: 
                idx_list_valid.append(idx_valid)
        
        idx_list = np.concatenate(idx_list)
        if self.validation is not None: 
            idx_list_valid = np.concatenate(idx_list_valid)
        
        def make_subset(dataset_copy, idx_list):
            if self.args.dataset in ['cifar100', 'cifar10', 'mnist', 'svhn']:
                dataset_copy.data = self.dataset.data[idx_list]
                dataset_copy.targets = self.dataset.targets[idx_list]
                dataset_copy.names = self.dataset.names[idx_list]
            elif self.args.dataset in ['imgnet380', 'timgnet']:
                dataset_copy = Subset(self.dataset, idx_list)
                # For convenience later, create data and names for Subset.
                dataset_copy.names = self.dataset.names[idx_list]
                dataset_copy.targets = self.dataset.targets[idx_list]
                dataset_copy.transform = self.dataset.transform
            else:
                raise NotImplementedError()
            return dataset_copy
        
        dataset_ = make_subset(dataset_, idx_list)
        if self.validation is not None:
            dataset_valid = make_subset(dataset_valid, idx_list_valid)

        if self.validation is None:
            return dataset_
        else:
            self.args.logger.print(f"******* Validation {self.validation} used *******")
            return dataset_, dataset_valid

def get_data(args):
    """Load dataset and relabel for different class_orders"""
    train_transform, test_transform = augmentations(args)

    if args.dataset != 'imgnet380':
        if os.path.isdir('/home/gyuhak'):
            args.root = '/home/gyuhak/data'
        elif os.path.isdir('/home/gkim87'):
            args.root = '/home/gkim87/data'
    else:
        if os.path.isdir('/home/gyuhak'):
            args.root = '/home/gyuhak/data'
        elif os.path.isdir('/home/gkim87'):
            args.root = '/home/share'

    class_split = ClassSplit(args)

    if args.dataset == 'mnist':
        train = MNIST(root=args.root, train=True, download=True, transform=train_transform)
        test  = MNIST(root=args.root, train=False, download=True, transform=test_transform)
    elif args.dataset == 'svhn':
        train = SVHN(root=args.root, split='train', download=True, transform=train_transform)
        test  = SVHN(root=args.root, split='test', download=True, transform=test_transform)
    elif args.dataset == 'cifar100':
        train = CIFAR100(root=args.root, train=True, download=True, transform=train_transform)
        test  = CIFAR100(root=args.root, train=False, download=True, transform=test_transform)
    elif args.dataset == 'cifar10':
        train = CIFAR10(root=args.root, train=True, download=True, transform=train_transform)
        test  = CIFAR10(root=args.root, train=False, download=True, transform=test_transform)
    elif args.dataset == 'imgnet380':
        train = ImageFolder(root=args.root + '/ImageNet/train', transform=train_transform)
        test = ImageFolder(root=args.root + '/ImageNet/val', transform=test_transform)

        samples, targets = [], []
        for pairs in train.samples:
            if pairs[1] in class_split.split:
                samples.append(pairs)
                targets.append(pairs[1])
        train.samples = samples
        train.targets = targets

        samples, targets = [], []
        for pairs in test.samples:
            if pairs[1] in class_split.split:
                samples.append(pairs)
                targets.append(pairs[1])
        test.samples = samples
        test.targets = targets

    elif args.dataset == 'timgnet':
        train = ImageFolder(root=args.root + '/Tiny_ImageNet/train', transform=train_transform)
        test = ImageFolder(root=args.root + '/Tiny_ImageNet/val_folders', transform=test_transform)

    train = class_split.relabel(train)
    test = class_split.relabel(test)

    return train, test