from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100 as CIFAR100_ORI

class CIFAR100(CIFAR100_ORI):
    """Modified from torchvision CIFAR100"""
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root, train, transform, target_transform, download)
        self.targets = np.array(self.targets)
        self.margin = np.zeros(len(self.targets))

        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            Dict:
        """
        img, target, names = self.data[index], self.targets[index], self.names[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img_transform = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return {'indices': index, 'inputs': img_transform, 
                'targets': target,
                'names': names, 'orig': self.not_aug_transform(img)}

class CIFAR10(CIFAR100):
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(CIFAR10, self).__init__(root, train, transform, target_transform, download)
        self.targets = np.array(self.targets)
