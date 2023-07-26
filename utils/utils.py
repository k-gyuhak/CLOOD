import os
import sys
import math
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset, Sampler, Subset
from datetime import datetime

# confusion matrix
from sklearn.metrics import confusion_matrix

# tsne
from sklearn.manifold import TSNE
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

def make_loader(data, args, batch_size, train='train', shuffle=True):
    if train == 'train':
        return DataLoader(data, batch_size=batch_size,
                            num_workers=args.num_workers,
                            pin_memory=args.pin_memory,
                            shuffle=shuffle)
    elif train == 'test':
        return DataLoader(data, batch_size=batch_size,
                            num_workers=args.num_workers,
                            pin_memory=args.pin_memory,
                            shuffle=False)
    else:
        raise NotImplementedError("'train' must be either train, calibration, or test")

class Criterion(nn.Module):
    def __init__(self, args, reduction='mean'):
        super(Criterion, self).__init__()
        self.args = args
        if args.loss_f == 'ce':
            args.logger.print("Using Cross-Entropy Loss")
            self.criterion = nn.CrossEntropyLoss(reduction=reduction)
        elif args.loss_f == 'bce':
            args.logger.print("Using Binary Cross-Entropy Loss")
            self.criterion = nn.BCELoss(reduction=reduction)
        elif args.loss_f == 'nll':
            args.logger.print("Using Negative Log-Likelihood Loss")
            self.criterion = nn.NLLLoss(reduction=reduction)
        else:
            raise NotImplementedError("Loss {} is not defined".format(args.loss_f))

    def forward(self, x, labels):
        labels = self.convert_lab(labels)
        if self.args.loss_f == 'bce':
            return self.criterion(torch.sigmoid(x), labels)
        elif self.args.loss_f == 'nll':
            return self.criterion(nn.LogSoftmax(dim=1)(x), labels)
        else: # 'ce'
            return self.criterion(x, labels)

    def convert_lab(self, labels):
        if self.args.loss_f == 'bce':
            raise NotImplementedError("BCE is not implemented")
            n_cls = len(self.seen_classes)
            labels = torch.eye(n_cls).to(self.args.device)[labels]
            return labels
        else: # 'ce', 'nll'
            return labels

class Logger:
    def __init__(self, args, name=None):
        self.init = datetime.now()
        self.args = args
        if name is None:
            self.name = self.init.strftime("%m|%d|%Y %H|%M|%S")
        else:
            self.name = name

        self.args.dir = self.name

        self._make_dir()

    def now(self):
        time = datetime.now()
        diff = time - self.init
        self.print(time.strftime("%m|%d|%Y %H|%M|%S"), f" | Total: {diff}")

    def print(self, *object, sep=' ', end='\n', flush=False, filename='/result.txt'):
        print(*object, sep=sep, end=end, file=sys.stdout, flush=flush)

        if self.args.print_filename is not None:
            filename = self.args.print_filename
        with open(self.dir() + filename, 'a') as f:
            print(*object, sep=sep, end=end, file=f, flush=flush)

    def _make_dir(self):
        # If provided hdd drive
        if 'hdd' in self.name or 'sdb' in self.name:
            if not os.path.isdir('/' + self.name):
                os.makedirs('/' + self.name)
        else:
            if not os.path.isdir(self.name):
                os.makedirs(self.name)
            # if not os.path.isdir(''):
            #     os.mkdir('')

    def dir(self):
        if 'hdd' in self.name or 'sdb' in self.name:
            return '/' + self.name + '/'
        else:
            return f'./{self.name}/'
            # './logs/{}/'.format(self.name)

    def time_interval(self):
        self.print("Total time spent: {}".format(datetime.now() - self.init))

class Tracker:
    def __init__(self, args):
        self.print = args.logger.print
        self.mat = np.zeros((args.n_tasks * 2 + 1, args.n_tasks * 2 + 1)) - 100

    def update(self, acc, task_id, p_task_id):
        """
            acc: float, accuracy
            task_id: int, current task id
            p_task_id: int, previous task's task id
        """
        self.mat[task_id, p_task_id] = acc

        # Compute average
        self.mat[task_id, -1] = np.mean(self.mat[task_id, :task_id + 1])

        # Compute forgetting
        for i in range(task_id):
            self.mat[-1, i] = self.mat[i, i] - self.mat[task_id, i]

        # Compute average incremental accuracy
        self.mat[-1, -1] = np.mean(self.mat[:task_id + 1, -1])

    def print_result(self, task_id, type='acc', print=None):
        if print is None: print = self.print
        if type == 'acc':
            # Print accuracy
            for i in range(task_id + 1):
                for j in range(task_id + 1):
                    acc = self.mat[i, j]
                    if acc != -100:
                        print("{:.2f}\t".format(acc), end='')
                    else:
                        print("\t", end='')
                print("{:.2f}".format(self.mat[i, -1]))
        elif type == 'forget':
            # Print forgetting and average incremental accuracy
            for i in range(task_id + 1):
                acc = self.mat[-1, i]
                if acc != -100:
                    print("{:.2f}\t".format(acc), end='')
                else:
                    print("\t", end='')
            print("{:.2f}".format(self.mat[-1, -1]))
            if task_id > 0:
                forget = np.mean(self.mat[-1, :task_id])
                print("{:.2f}".format(forget))
        else:
            raise NotImplementedError("Type must be either 'acc' or 'forget'")

class AUCTracker:
    def __init__(self, args):
        self.print = args.logger.print
        self.mat = np.zeros((args.n_tasks * 2 + 1, args.n_tasks * 2 + 1)) - 100
        self.n_tasks = args.n_tasks

    def update(self, acc, task_id, p_task_id):
        """
            acc: float, accuracy
            task_id: int, current task id
            p_task_id: int, previous task's task id
        """
        self.mat[task_id, p_task_id] = acc

        # Compute average
        self.mat[task_id, -1] = np.mean(np.concatenate([
            self.mat[task_id, :task_id],
            self.mat[task_id, task_id + 1:self.n_tasks]
        ]))

    def print_result(self, task_id, type='acc', print=None):
        # Compute average incremental accuracy
        self.mat[-1, -1] = np.mean(self.mat[:task_id + 1, -1])
        if print is None: print = self.print
        if type == 'acc':
            # Print accuracy
            for i in range(task_id + 1):
                for j in range(self.n_tasks):
                    acc = self.mat[i, j]
                    if acc != -100:
                        print("{:.2f}\t".format(acc), end='')
                    else:
                        print("\t", end='')
                print("{:.2f}".format(self.mat[i, -1]))
            # Print forgetting and average incremental accuracy
            for i in range(self.n_tasks):
                print("\t", end='')
            print("{:.2f}".format(self.mat[-1, -1]))
        else:
            raise NotImplementedError("Type must be 'acc'")

class OWTracker:
    def __init__(self, args):
        self.print = args.logger.print
        self.mat = np.zeros((args.n_tasks * 2 + 1, args.n_tasks * 2 + 1)) - 100
        self.n_tasks = args.n_tasks

    def update(self, acc, task_id, p_task_id):
        """
            acc: float, accuracy
            task_id: int, current task id
            p_task_id: int, previous task's task id
        """
        self.mat[task_id, p_task_id] = acc

        # Compute average
        self.mat[task_id, -1] = np.mean(self.mat[task_id, task_id + 1:self.n_tasks])

        # Compute average incremental accuracy
        self.mat[-1, -1] = np.mean(self.mat[:task_id + 1, -1])

    def print_result(self, task_id, type='acc', print=None):
        if print is None: print = self.print
        if type == 'acc':
            # Print accuracy
            for i in range(task_id + 1):
                for j in range(self.n_tasks):
                    acc = self.mat[i, j]
                    if acc != -100:
                        print("{:.2f}\t".format(acc), end='')
                    else:
                        print("\t", end='')
                if self.mat[i, -1] != -100:
                    print("{:.2f}".format(self.mat[i, -1]))
                else:
                    print("")
            # Print forgetting and average incremental accuracy
            for i in range(self.n_tasks):
                print("\t", end='')
            print("{:.2f}".format(self.mat[-1, -1]))
        else:
            raise NotImplementedError("Type must be 'acc'")

def print_result(mat, task_id, type, print=print):
    if type == 'acc':
        # Print accuracy
        for i in range(task_id + 1):
            for j in range(task_id + 1):
                acc = mat[i, j]
                if acc != -100:
                    print("{:.2f}\t".format(acc), end='')
                else:
                    print("\t", end='')
            print("{:.2f}".format(mat[i, -1]))
    elif type == 'forget':
        # Print forgetting and average incremental accuracy
        for i in range(task_id + 1):
            acc = mat[-1, i]
            if acc != -100:
                print("{:.2f}\t".format(acc), end='')
            else:
                print("\t", end='')
        print("{:.2f}".format(mat[-1, -1]))
        if task_id > 0:
            forget = np.mean(mat[-1, :task_id])
            print("Average Forgetting: {:.2f}".format(forget))
    else:
        ValueError("Type must be either 'acc' or 'forget'")

def tsne(train_f_cross, train_y_cross, name='tsne',
         n_components=2, verbose=0, learning_rate=1, perplexity=9, n_iter=1000, logger=None):
    """ train_f_cross: X, numpy array. train_y_cross: y, numpy array """
    num_y = len(list(set(train_y_cross)))

    tsne = TSNE(n_components=n_components, verbose=verbose,
                learning_rate=learning_rate, perplexity=perplexity,
                n_iter=n_iter)
    tsne_results = tsne.fit_transform(train_f_cross)

    df_subset = pd.DataFrame(data={'tsne-2d-one': tsne_results[:, 0],
                                    'tsne-2d-two': tsne_results[:, 1]})
    df_subset['y'] = train_y_cross

    plt.figure(figsize=(16,10))
    sn.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=sn.color_palette("hls", num_y),
        data=df_subset,
        legend="full",
        alpha=0.3
    )

    dir = '' if logger is None else logger.dir()

    plt.savefig(dir + name)
    plt.close()

def plot_confusion(true_lab, pred_lab, label_names=None,
                    task_id=None, p_task_id=None, name='confusion',
                    print=print, logger=None, n_cls_task=None,
                    second_label_name=None):
    """
    If second_label_name is provided, create a duplicate confusion matrix with the provided second_label_name
    """
    print = logger.print if logger is not None else print
    dir = '' if logger is None else logger.dir() # if None, save into current folder

    classes = sorted(set(np.concatenate([true_lab, pred_lab])))
    if label_names is not None:
        labs = []
        for c in classes:
            labs.append(label_names[c])
    plt.figure(figsize=(15, 14))
    cm = confusion_matrix(true_lab, pred_lab)
    np.savetxt(os.path.join(dir, f'cm_{task_id}.txt'), cm, fmt='%i')
    hmap = sn.heatmap(cm, annot=True)
    hmap.set_xticks(np.arange(len(classes)) + 0.5)
    hmap.set_yticks(np.arange(len(classes)) + 0.5)
    if label_names is not None:
        hmap.set_xticklabels(labs, rotation=90)
        hmap.set_yticklabels(labs, rotation=0)

    if n_cls_task is not None:
        for j in range(len(classes)):
            if (j + 1) % n_cls_task == 0:
                plt.axhline(y=j + 1)
                plt.axvline(x=j + 1)

    if task_id is not None:
        plt.savefig(dir + f"Total Task {task_id}, current task {p_task_id} is learned.pdf",
                    bbox_inches='tight')
    else:
        plt.savefig(dir + name + '.pdf', bbox_inches='tight')

    if second_label_name is not None:
        if second_label_name is not None:
            labs = []
            for c in classes:
                labs.append(second_label_name[c])
            hmap.set_xticklabels(labs, rotation=90)
            hmap.set_yticklabels(labs, rotation=0)
            if task_id is not None:
                plt.savefig(dir + f"Total Task {task_id}, current task {p_task_id} is learned (second label).pdf",
                            bbox_inches='tight')
            else:
                plt.savefig(dir + name + ' (second label).pdf', bbox_inches='tight')
            plt.close()

    if task_id is not None:
        print("{}/{} | upper/lower triangular sum: {}/{}".format(task_id, p_task_id,
                                    np.triu(cm, 1).sum(), np.tril(cm, -1).sum()))
    else:
        print("Upper/lower triangular sum: {}/{}".format(np.triu(cm, 1).sum(),
                                                        np.tril(cm, -1).sum()))

def dist_estimation(data, classes):
    # data, classes: numpy array
    data = data / np.linalg.norm(data, axis=-1, keepdims=True)

    unique_cls = list(sorted(set(classes)))

    mu_list = []
    sigma_list = []
    for i, c in enumerate(unique_cls):
        idx = classes == c
        selected_data = data[idx]

        mu = np.mean(selected_data, axis=0)
        mu_list.append(mu)

        sigma = 0
        selected_data = selected_data - mu
        for s in selected_data:
            s = s.reshape(1, -1)
            sigma += np.transpose(s) @ s
        sigma_list.append(sigma / len(selected_data))
    # sigma /= len(data)
    return mu_list, sigma_list

def maha_distance(inputs, mu, sigma):
    inv_sigma = np.linalg.inv(sigma)
    out = (inputs - mu).dot(inv_sigma)
    out = np.sum(out * (inputs - mu), axis=1)
    return out

def md(data, mean, mat, inverse=False):
    if isinstance(data, torch.Tensor):
        data = data.data.cpu().numpy()
    if data.ndim == 1:
        data.reshape(1, -1)
    delta = (data - mean)

    if not inverse:
        mat = np.linalg.inv(mat)

    dist = np.dot(np.dot(delta, mat), delta.T)
    return np.sqrt(np.diagonal(dist)).reshape(-1, 1)

from sklearn.metrics import roc_auc_score
def compute_auc(in_scores, out_scores):
    """
        in_scores: np.array of shape (N, 1)
        out_scores: np.array of shape (M, 1)

        It returns auc e.g., auc=0.95
    """
    if isinstance(in_scores, list):
        in_scores = np.concatenate(in_scores)
    if isinstance(out_scores, list):
        out_scores = np.concatenate(out_scores)

    labels = np.concatenate([np.ones_like(in_scores),
                             np.zeros_like(out_scores)])
    try:
        auc = roc_auc_score(labels, np.concatenate((in_scores, out_scores)))
    except ValueError:
        print("Input contains NaN, infinity or a value too large for dtype('float64').")
        auc = -0.99
    return auc

def auc(score_dict, task_id, auc_tracker):
    """
        score_dict: is a dictionary (k, v), where
        k is the task id of a test data and v, np.array of shape (K, T), is the corresponding scores

        AUC: AUC_ij = output values of task i's heads using i'th task data (IND)
                      vs output values of task i's head using j'th task data (OOD)
        NOTE 
    """
    in_scores = score_dict[task_id][:, task_id]

    for k, val in score_dict.items():
        if k != task_id:
            ood_scores = val[:, task_id]
            auc_value = compute_auc(in_scores, ood_scores)
            auc_tracker.update(auc_value * 100, task_id, k)
    return auc_tracker.mat[task_id, :len(score_dict)]

def auc_cil(score_dict, task_id, last_task_id, auc_tracker):
    """
        AUC by CIL style.
        score_dict: {data_id: np.array, size (K, T), ...},
                    where K is the sample size of data_id's data and T is the number of tasks.
                    data_id ranges from 0 to T-1
        last_task_id: last learned task id. Since it's CIL style AUC,
                      it does not make sense to compare IND score against OOD score of unlearned task network
                      Previously, in AUC(), last_task_id wasn't necessary since we don't use unlearned task network's output value
        AUC_ij = output values of task i's heads using i'th task data (IND)
                 vs output values of task j's head using i'th task data (OOD)
    """
    in_scores = score_dict[task_id][:, task_id]

    scores = score_dict[task_id]
    for k in range(last_task_id + 1):
        if k != task_id:
            ood_scores = scores[:, k]
            auc_value = compute_auc(in_scores, ood_scores)
            auc_tracker.update(auc_value * 100, task_id, k)
    return auc_tracker.mat[task_id, :len(score_dict)]

def auc_ow(score_dict, last_task_id, auc_tracker):
    in_scores = []
    for k in range(last_task_id + 1):
        scores = score_dict[k][:, :last_task_id + 1]
        scores = np.max(scores, axis=1)
        in_scores.append(scores)
    in_scores = np.concatenate(in_scores)

    for k in range(last_task_id + 1, len(score_dict)):
        scores = score_dict[k][:, :last_task_id + 1]
        ood_scores = np.max(scores, axis=1)

        auc_value = compute_auc(in_scores, ood_scores)
        auc_tracker.update(auc_value * 100, last_task_id, k)
    return auc_tracker.mat[last_task_id, (last_task_id + 1):len(score_dict)]

def auc_output_ow(output_dict, last_task_id, auc_tracker, n_cls_task):
    """
        output_dict: dictionary whose keys are all the task_ids.
                    The value of each key is a tensor of size (N_t, C_t),
                    where N_t is the number of samples of task t
                    and C_t is the number of classes learned until task t.
    """
    score_dict = {}
    for k, v in output_dict.items():
        b, c = v.shape
        score_dict[k] = np.max(v.reshape(b, -1, n_cls_task), axis=-1)
    out = auc_ow(score_dict, last_task_id, auc_tracker)
    return out

def auc_output(output_dict, task_id, auc_tracker, n_cls_task):
    """
        Compute AUC for the provided outputs.
        The difference between this and auc() is auc() receives dict of scores
        while this one receives dict of outputs before scores.
        Score of a sample is usually the maximum of output.
        arguments:
            Output_dict is a dictionary with (key, value) pairs.
            key is the task_id, and value is the corresponding outputs in np.array
    """
    score_dict = {}
    for k, v in output_dict.items():
        b, c = v.shape
        score_dict[k] = np.max(v.reshape(b, -1, n_cls_task), axis=-1)
    out = auc(score_dict, task_id, auc_tracker)
    return out

def auc_output_cil(output_dict, task_id, auc_tracker, n_cls_task):
    """
        Same as auc_output(), but returns auc_cil.
    """
    score_dict = {}
    for k, v in output_dict.items():
        b, c = v.shape
        score_dict[k] = np.max(v.reshape(b, -1, n_cls_task), axis=-1)
    out = auc_cil(score_dict, task_id, auc_tracker)
    return out

class DeNormalize(object):
    # def __init__(self, mean, std):
    def __init__(self, transform):
        self.mean = transform.transforms[-1].mean # (Tensor)
        self.std = transform.transforms[-1].std # (Tensor)

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

class MySampler(Sampler):
    """
        Sampler for dataset whose length is different from that of the target dataset.
        This can be useful when we need oversampling/undersampling because
        the target dataset has more/less samples than the dataset of interest.
        Generate indices whose length is same as that of the target length * maximum number of epochs.
    """
    def __init__(self, current_length, target_length, batch_size, max_epoch):
        self.current = current_length       
        self.length = math.ceil(target_length / batch_size) * batch_size * max_epoch

    def __iter__(self):
        self.indices = np.array([], dtype=int)
        while len(self.indices) < self.length:
            idx = np.random.permutation(self.current)
            self.indices = np.concatenate([self.indices, idx])
        self.indices = self.indices[:self.length]
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

class Memory(Dataset):
    """
        Replay buffer. Keep balanced samples. Data must be compatible with Image.
        Currently, MNIST and CIFAR are compatible.
    """
    def __init__(self, args):
        self.args = args
        self.buffer_size = args.buffer_size

        self.data_list = {}
        self.targets_list = {}

        self.data, self.targets = [], []

        self.n_cls = len(self.data_list)
        self.n_samples = self.buffer_size

    def update(self, dataset):
        self.args.logger.print("Updating Memory")
        self.transform = dataset.transform

        ys = list(sorted(set(dataset.targets)))
        for y in ys:
            idx = np.where(dataset.targets == y)[0]
            self.data_list[y] = dataset.data[idx]
            self.targets_list[y] = dataset.targets[idx]

            self.n_cls = len(self.data_list)

        self.n_samples = self.buffer_size // self.n_cls
        for y, data in self.data_list.items():
            idx = np.random.permutation(len(data))
            idx = idx[:self.n_samples]
            self.data_list[y] = self.data_list[y][idx]
            self.targets_list[y] = self.targets_list[y][idx]

        self.data, self.targets = [], []
        for (k, data), (_, targets) in zip(self.data_list.items(), self.targets_list.items()):
            self.data.append(data)
            self.targets.append(targets)
        self.data = np.concatenate(self.data)
        self.targets = np.concatenate(self.targets)

    def is_empty(self):
        return len(self.data) == 0

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)

class Memory_ImageFolder(Dataset):
    """
        Replay buffer. Keep balanced samples. This only works for ImageFolder dataset.
    """
    def __init__(self, args):
        self.args = args
        self.buffer_size = args.buffer_size

        self.data_dict = defaultdict(np.array)
        self.targets_dict = defaultdict(np.array)
        self.indices_dict = defaultdict(np.array)

        self.data, self.targets, self.indices = [], [], []

        self.n_cls = len(self.data_dict)
        self.n_samples = self.buffer_size

    def is_empty(self):
        return len(self.data) == 0

    def update(self, dataset):
        """
        Update the memory buffer
        
        Args:
            dataset: it's an instance of pytorch Subset. The subset has attributes targets, indices, transform, etc.
            where indices are the absolute indices in the original dataset
            and targets are the targets of the indices in the original dataset.
            This function makes attributes data and targets, where
            data is an np.array of paths and targets is an np.array of targets of the corresponding data

            NOTE
            dataset is a Subset
            dataset.dataset is an ImageFolder
        """
        self.args.logger.print("Updating Memory")

        self.loader = dataset.dataset.loader
        self.transform = dataset.dataset.transform
        ys = list(sorted(set(dataset.targets)))
        for y in ys:
            idx = np.where(dataset.targets == y)[0]
            absolute_idx = dataset.indices[idx]
            self.data_dict[y] = np.array([dataset.dataset.samples[i][0] for i in absolute_idx])
            self.targets_dict[y] = np.array([dataset.dataset.samples[i][1] for i in absolute_idx])
            self.indices_dict[y] = absolute_idx

        self.n_cls = len(self.data_dict) # total number of classes in memory

        # number of samples per class
        self.n_samples = self.buffer_size // self.n_cls
        for y, data in self.data_dict.items():
            # Choose random samples to keep
            idx = np.random.permutation(len(data))
            idx = idx[:self.n_samples]
            self.data_dict[y] = self.data_dict[y][idx]
            self.targets_dict[y] = self.targets_dict[y][idx]
            self.indices_dict[y] = self.indices_dict[y][idx]

        self.data, self.targets, self.indices = [], [], []
        for (k, data), (_, targets), (_, idx) in zip(self.data_dict.items(), self.targets_dict.items(), self.indices_dict.items()):
            assert len(data) == len(targets)
            assert len(targets) == len(idx)
            self.data.append(data)
            self.targets.append(targets)
            self.indices.append(idx)
        self.data = np.concatenate(self.data)
        self.targets = np.concatenate(self.targets)
        self.indices = np.concatenate(self.indices)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.data[index], self.targets[index]
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)

def load_pretrain(args, target_model, n_pre_cls=None):
    """
        target_model: the model we want to replace the parameters (most likely un-trained)
    """
    if n_pre_cls is None:
        args.logger.print("Loading Deit model pre-trained with 611 classes")
        if os.path.isfile('./deit_pretrained/best_checkpoint.pth'):
            checkpoint = torch.load('./deit_pretrained/best_checkpoint.pth', map_location='cpu')
        else:
            raise NotImplementedError("Cannot find pre-trained model")
    elif n_pre_cls == 200:
        args.logger.print("Loading Deit model pre-trained with randomly selected 200 classes")
        if os.path.isfile('./deit_pretrained_200/best_checkpoint.pth'):
            checkpoint = torch.load('./deit_pretrained_200/best_checkpoint.pth', map_location='cpu')
        else:
            raise NotImplementedError("Cannot find pre-trained model")
            
    target = target_model.state_dict()
    pretrain = checkpoint['model']
    transfer, missing = {}, []
    for k, _ in target.items():
        if k in pretrain and 'head' not in k:
            transfer[k] = pretrain[k]
        else:
            missing.append(k)
    target.update(transfer)
    target_model.load_state_dict(target)
    args.logger.print("Parameters not updated: ", missing)
    return target_model

class ComputeEnt:
    def __init__(self, args):
        self.temp = args.T

    def compute(self, output, keepdim=True):
        """
            output: torch.tensor logit, 2d
        """
        soft = torch.softmax(output, dim=-1)
        if keepdim:
            return -1 * torch.sum(soft * torch.log(soft), dim=-1, keepdim=True)
        else:
            return-1 * torch.sum(soft * torch.log(soft))

def custom_load_state_dict(args, model, checkpoint_dict):
    model_dict = model.state_dict()

    pretrained_dict = {k: v for k, v in checkpoint_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
    model_dict.update(pretrained_dict)

    model.load_state_dict(model_dict)

    not_updated = []
    for k in model_dict:
        if k not in pretrained_dict:
            not_updated.append(k)
    args.logger.print("The following parameters are not updated:")
    args.logger.print(not_updated)
    args.logger.print()

def custom_load(resume_path, map_location="cuda", server='s145', file_type='torch'):
    try:
        from_ssh = False
        if file_type == 'torch':
            checkpoint = torch.load(resume_path)
        elif file_type == 'numpy':
            checkpoint = np.load(resume_path)
        else:
            raise NotImplementedError()
    except FileNotFoundError:
        raise NotImplementedError()
    return checkpoint