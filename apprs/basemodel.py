import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR

from utils.utils import *

class BaseModel(nn.Module):
    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.scores, self.scores_md, self.scores_total, self.out_score = [], [], [], []
        self.n_cls_task = args.n_cls_task
        
        self.buffer = None

        self.args = args
        self.criterion = args.criterion # NLL
        self.net = args.net

        self.scheduler = None

        self.correct, self.til_correct, self.total, self.total_loss = 0., 0., 0., 0.
        self.output_list, self.feature_list, self.label_list, self.pred_list = [], [], [], []
        self.md_output_save = []
        self.md_ow_list = []

        self.saving_buffer = {}

    def observe(self, inputs, labels, not_aug_inputs=None, f_y=None, **kwargs):
        pass

    def evaluate(self, inputs, labels, task_id, **kwargs):
        total_learned_task_id = kwargs['total_learned_task_id']
        self.net.eval()

        with torch.no_grad():
            features = self.net.forward_features(inputs)
            out = self.net.forward_classifier(features)

        self.feature_list.append(features.data.cpu().numpy())
        out = F.softmax(out / self.args.T_odin, dim=-1)
        pred = out.argmax(1)

        # CIL
        scores, pred = out[:, :(total_learned_task_id + 1) * self.n_cls_task].max(1)
        self.scores_total.append(scores.detach().cpu().numpy())
        self.correct += pred.eq(labels).sum().item()
        self.total += len(labels)

        # TIL
        normalized_labels = labels % self.n_cls_task
        til_out = out[:, task_id * self.n_cls_task:(task_id + 1) * self.n_cls_task]
        scores, til_pred = til_out.max(1)
        self.scores.append(scores.detach().cpu().numpy())
        self.til_correct += til_pred.eq(normalized_labels).sum().item()

        self.net.train()

        self.pred_list.append(pred.cpu().numpy())

        self.output_list.append(out.data.cpu().numpy())
        self.label_list.append(labels.data.cpu().numpy())

    def end_epoch(self):
        if self.scheduler is not None:
            self.scheduler.step()

    def acc(self, reset=True):
        metrics = {}
        metrics['cil_acc'] = self.correct / self.total * 100
        metrics['til_acc'] = self.til_correct / self.total * 100
        if len(self.label_list) > 0: metrics['label_list'] = np.concatenate(self.label_list)
        if len(self.pred_list) > 0: metrics['pred_list'] = np.concatenate(self.pred_list)
        if len(self.output_list) > 0: metrics['output_list'] = np.concatenate(self.output_list)
        if len(self.md_ow_list) > 0: metrics['md_ow_list'] = np.concatenate(self.md_ow_list)
        if len(self.scores) > 0: metrics['scores'] = np.concatenate(self.scores)
        if len(self.md_output_save) > 0: metrics['md_list']  = np.concatenate(self.md_output_save)
        if reset: self.reset_eval()
        return metrics

    def reset_eval(self):
        self.correct, self.til_correct, self.total, self.total_loss = 0., 0., 0., 0.
        self.output_list, self.label_list, self.pred_list = [], [], []
        self.scores, self.scores_md, self.scores_total = [], [], []
        self.feature_list, self.label_list = [], []
        self.md_output_save, self.md_ow_list = [], []

    def save(self, task_id, **kwargs):
        """
            Save model specific elements required for resuming training
            kwargs: e.g. model state_dict, optimizer state_dict, epochs, etc.
        """
        raise NotImplementedError()

    def load(self, **kwargs):
        raise NotImplementedError()

    def compute_stats(self, task_id, loader):
        self.args.logger.print(f"Compute stats for task {task_id}")
        self.reset_eval()
        self.net.eval()

        for data in loader:
            inputs = data['inputs'].to(self.args.device)
            labels = data['targets'].to(self.args.device)

            self.evaluate(
                inputs,
                labels,
                task_id,
                total_learned_task_id=task_id,
            )

        self.feature_list = np.concatenate(self.feature_list)
        self.label_list = np.concatenate(self.label_list)

        torch.save(
            self.feature_list,
            self.args.logger.dir() + f'/feature_task_{task_id}',
        )
        torch.save(
            self.label_list,
            self.args.logger.dir() + f'/label_task_{task_id}'
        )

        cov_list = []
        ys = list(sorted(set(self.label_list)))
        for y in ys:
            idx = np.where(self.label_list == y)[0]
            f = self.feature_list[idx]
            
            mean = np.mean(f, 0)
            self.args.mean[y] = mean
            np.save(
                os.path.join(self.args.logger.dir(), f'{self.args.mean_label_name}_{y}'),
                mean
            )

            cov = np.cov(f.T)
            cov_list.append(cov)
        cov = np.array(cov_list).mean(0)
        self.args.cov[task_id] = cov
        self.args.cov_inv[task_id] = np.linalg.inv(0.8 * cov + 0.2 * np.eye(len(cov)))
        np.save(
            os.path.join(self.args.logger.dir(), f'{self.args.cov_task_name}_{task_id}'),
            cov
        )

        self.net.train()
        self.reset_eval()

    def compute_md_by_task(self, net_id, features):
        """
            Compute Mahalanobis distance of features to the Gaussian distribution of task == net_id
            return: scores_md of np.array of ndim == (B, 1) if cov_inv is available
                    None if cov_inv is not available (e.g. task=0 or cov_inv is not yet computed)
        """
        md_list, dist_list = [], []
        if len(self.args.cov_inv) >= net_id + 1:
            for y in range(net_id * self.n_cls_task, (net_id + 1) * self.n_cls_task):
                mean, cov_inv = self.mean_cov(y, net_id)
                dist = md(features, mean, cov_inv, inverse=True)

                scores_md = 1 / dist
                md_list.append(scores_md)
                dist_list.append(-dist)

            scores_md = np.concatenate(md_list, axis=1)
            dist_list = np.concatenate(dist_list, axis=1)
            scores_md = scores_md.max(1, keepdims=True)
            dist_list = dist_list.max(1)
            return scores_md, dist_list
        return None, None

    def mean_cov(self, y, net_id, inverse=True):
        if inverse:
            cov = self.args.cov_inv[net_id]
        else:
            cov = self.args.cov[net_id]
        return self.args.mean[y], cov