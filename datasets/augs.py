from copy import deepcopy
import torchvision.transforms as transforms
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

def augmentations(args):
    if 'resnet' in args.architecture or 'alexnet' in args.architecture:
        args.logger.print('Using the standard augmentations (e.g. ResNet, AlexNet, etc.)')
        if args.dataset == 'mnist':
            args.mean = (0.1307, 0.1307, 0.1307)
            args.std = (0.3081, 0.3081, 0.3081)
            train_transform = transforms.Compose([
                transforms.RandomCrop(28, padding=2),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize((0.1307), (0.3081))
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307), (0.3081))
            ])
        elif args.dataset == 'cifar10':
            args.mean = (0.4914, 0.4822, 0.4465)
            args.std = (0.247, 0.243, 0.261)
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(args.mean, args.std),
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(args.mean, args.std),
            ])
        elif args.dataset == 'cifar100':
            args.mean = (0.5071, 0.4866, 0.4409)
            args.std = (0.2009, 0.1984, 0.2023)
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.24705882352941178),
                #transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(args.mean, args.std),
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(args.mean, args.std),
            ])
        # The tiny-imagenet data is resized to 32 as CIFAR so that we can use the same architecture we used for CIFAR
        elif args.dataset == 'timgnet':
            args.mean = (0.485, 0.456, 0.406)
            args.std = (0.229, 0.224, 0.225)
            train_transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(args.mean, args.std),
                ])
            test_transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize(args.mean, args.std),
                    ])
        else:
            raise NotImplementedError()
        return train_transform, test_transform
    elif 'deit' in args.architecture or 'vit' in args.architecture:
        args.logger.print('Using augmentations of ViT')
        model_ = timm.create_model(args.architecture, pretrained=False, num_classes=1).cuda()
        config = resolve_data_config({}, model=model_)
        TRANSFORM = create_transform(**config)

        test_transform = deepcopy(TRANSFORM)
        return TRANSFORM, test_transform
    else:
        raise NotImplementedError()