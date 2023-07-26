import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
from utils.sgd_hat import SGD_hat as SGD
from apprs.hat import HAT
from collections import Counter
from copy import deepcopy
from utils.utils import *
from torch.utils.data import DataLoader
from collections import Counter

device = "cuda" if torch.cuda.is_available() else "cpu"

class ROW(HAT):
    def __init__(self, args):
        super().__init__(args)
        self.p_mask, self.mask_back = None, None
        self.last_task_id = -1 # task id of lastly learned task

        if args.use_buffer:
            assert args.buffer_size
            if self.args.dataset in ['imagenet', 'timgnet']:
                self.buffer_dataset = Memory_ImageFolder(args)
            else:
                self.buffer_dataset = Memory(args)
        else:
            self.buffer_dataset = None

        self.ent = ComputeEnt(self.args)

    def observe(self, inputs, labels, names, orig, **kwargs):
        task_id, b, B = kwargs['task_id'], kwargs['b'], kwargs['B']
        s = self.update_s(b, B)

        n_samples = len(inputs)
        normalized_labels = labels % self.n_cls_task

        if self.buffer:
            inputs_bf, labels_bf = next(self.buffer_iter)
            inputs_bf = inputs_bf.to(device)
            
            labels_bf = torch.zeros_like(labels_bf).to(device) + self.n_cls_task
            normalized_labels_bf = labels_bf
            inputs = torch.cat([inputs, inputs_bf])
            labels = torch.cat([labels, labels_bf])
            normalized_labels = torch.cat([normalized_labels, normalized_labels_bf])

        with torch.cuda.amp.autocast(enabled=True):
            features, masks = self.net.forward_features(task_id, inputs, s=s)
            outputs = self.net.forward_classifier(task_id, features)
            loss = self.criterion(outputs, normalized_labels)
            loss += self.hat_reg(self.p_mask, masks)

        self.optimizer.zero_grad()
        # loss.backward()
        self.scaler.scale(loss).backward()
        self.compensation(self.net, self.args.thres_cosh, s=s)

        hat = False
        if self.last_task_id >= 0:
            hat = True
        # self.optimizer.step(hat=hat)
        self.scaler.step(self.optimizer, hat=hat)
        self.scaler.update()
        
        self.compensation_clamp(self.net, self.args.thres_emb)

        self.total_loss += loss.item()
        outputs = outputs[:n_samples]
        scores, pred = outputs.max(1)
        self.scores.append(scores.detach().cpu().numpy())
        self.correct += pred.eq(normalized_labels[:n_samples]).sum().item()
        self.total += n_samples

        return loss.item()

    def observe_ood(self, inputs, labels, **kwargs):
        task_id = kwargs['task_id']
        normalized_labels = labels % self.n_cls_task

        inputs_bf, labels_bf = next(self.buffer_iter)
        inputs_bf = inputs_bf.to(device)
        labels_bf = labels_bf.to(device)
        
        labels_bf_ood = torch.zeros_like(labels_bf) + self.n_cls_task
        inputs = torch.cat([inputs, inputs_bf])
        normalized_labels = torch.cat([normalized_labels, labels_bf_ood])

        with torch.no_grad():
            features, _ = self.net.forward_features(task_id, inputs, s=self.args.smax)
        outputs = self.net.tp_head[task_id](features)

        loss = self.criterion(outputs, normalized_labels)

        self.optimizer_clf.zero_grad()
        loss.backward()
        self.optimizer_clf.step()

        return loss.item()

    def preprocess_finetune_ood(self, **kwargs):
        task_id = kwargs['task_id']

        assert self.buffer_dataset is not None

        if len(self.net.tp_head) < task_id + 1:
            self.net.tp_head.append(deepcopy(self.net.head[task_id]))
            assert self.net.tp_head[task_id].weight.shape[0] == (self.args.n_cls_task + 1)

        self.optimizer_clf = SGD(
            self.net.tp_head[task_id].parameters(),
            lr=self.args.lr,
            momentum=self.args.momentum
        )

        self.sampler = MySampler(
            len(self.buffer_dataset),
            len(kwargs['loader'].dataset),
            self.args.batch_size_finetune,
            kwargs['n_epochs'],
        )
        
        self.buffer = DataLoader(
            self.buffer_dataset,
            batch_size=self.args.batch_size_finetune,
            sampler=self.sampler,
            num_workers=5,
            pin_memory=self.args.pin_memory
        )
        self.buffer_iter = iter(self.buffer)

    def observe_wp(self, inputs, labels, **kwargs):
        task_id = kwargs['task_id']

        n_samples = len(inputs)
        normalized_labels = labels % self.n_cls_task

        with torch.no_grad():
            features, _ = self.net.forward_features(task_id, inputs, s=self.args.smax)
        outputs = self.net.head[task_id](features)

        loss = self.criterion(outputs, normalized_labels)

        self.optimizer_clf.zero_grad()
        loss.backward()
        self.optimizer_clf.step()

        return loss.item()

    def preprocess_finetune_wp(self, **kwargs):
        task_id = kwargs['task_id']

        self.optimizer_clf = SGD(self.net.head[task_id].parameters(), lr=self.args.lr, momentum=self.args.momentum)

    def evaluate(self, inputs, labels, task_id, true_id=None, **kwargs):
        """
        Evaluate the model for both TIL and CIL. Prepare and save outputs for various purposes
        
        Args:
            total_learned_task_id: int, the last task_id the model has learned so far
        """

        total_learned_task_id = kwargs['total_learned_task_id']
        self.net.eval()
        self.total += len(labels)

        self.label_list.append(labels.data.cpu())

        normalized_labels = labels % self.n_cls_task

        out_list, output_ood = [], []
        with torch.no_grad():
            entropy_list, md_score_list, logit_output = [], [], []
            for t in range(total_learned_task_id + 1):
                features, _ = self.net.forward_features(t, inputs, s=self.args.smax)

                out = self.net.forward_classifier(t, features)[:, :self.args.n_cls_task]

                if t == task_id:
                    self.feature_list.append(features.data.cpu().numpy())
                    _, til_pred = out.max(1)
                    self.til_correct += til_pred.eq(normalized_labels).sum().item()

                out = F.softmax(out / self.args.T, dim=1)

                if self.args.compute_md:
                    scores, _ = self.compute_md_by_task(t, features)
                    if scores is not None: md_score_list.append(scores)

                if self.args.use_two_heads:
                    out_ood = self.net.tp_head[t](features)
                    out_ood = F.softmax(out_ood / self.args.T, dim=1)
                    out_ood = out_ood[:, :self.n_cls_task]
                    out = out * torch.max(out_ood, dim=-1, keepdims=True)[0]

                if self.args.task_inference == 'entropy':
                    entropy_list.append(self.ent.compute(out))
                else:
                    pass
                
                logit_output.append(out.data)

                out_list.append(out)
                output_ood.append(out)

        if len(entropy_list) > 0:
            entropy_list = torch.cat(entropy_list, dim=-1)
            task_id_pred = torch.min(entropy_list, dim=-1)[1]

        out_list = torch.cat(out_list, dim=1)
        output_ood = torch.cat(output_ood, dim=1)
        logit_output = torch.cat(logit_output, dim=1)

        self.output_list.append(logit_output.data.cpu().numpy())

        if len(md_score_list) > 0:
            md_score_list = np.concatenate(md_score_list, axis=1)
            self.md_output_save.append(md_score_list)

            md_score_list = torch.from_numpy(md_score_list)
            if self.args.use_md and total_learned_task_id + 1 == md_score_list.size(1):
                md_score_list = md_score_list.to(self.args.device).unsqueeze(-1)
                md_score_list = md_score_list / md_score_list.sum(dim=1, keepdims=True)

                out_list = out_list.view(out_list.size(0), total_learned_task_id + 1, -1) * md_score_list
                out_list = out_list.view(out_list.size(0), -1)

        if len(entropy_list) > 0:
            # check if task_id_pred are correct
            true_tasks = labels // self.n_cls_task
            idx = task_id_pred == true_tasks

            # consider samples correctly predicted
            if sum(idx) == 0:
                self.correct += 0
            else:
                _, pred_cor = out_list[idx].max(1)
                self.correct += pred_cor.eq(labels[idx]).sum().item()

            task_output_ood = []
            for task_pred, sample in zip(task_id_pred, output_ood):
                task_output_ood.append(sample[task_pred * self.n_cls_task:(task_pred + 1) * self.n_cls_task].view(1, -1))
            output_ood = torch.cat(task_output_ood)
            total_scores, _ = output_ood.max(1)
            self.scores_total.append(total_scores.detach().cpu().numpy())
        else:
            _, pred = out_list.max(1)
            self.correct += pred.eq(labels).sum().item()
            self.pred_list.append(pred.cpu().numpy())

            total_scores, _ = output_ood.max(1)
            self.scores_total.append(total_scores.detach().cpu().numpy())

        self.net.train()

    def save(self, task_id, **kwargs):
        """
            Save model-specific elements required for training
            kwargs: e.g. model state_dict, optimizer state_dict, epochs, etc.
        """
        self.saving_buffer['buffer_dataset'] = self.buffer_dataset
        self.saving_buffer['p_mask'] = self.p_mask
        self.saving_buffer['mask_back'] = self.mask_back
        self.saving_buffer['last_task_id'] = self.last_task_id

        for key in kwargs:
            self.saving_buffer[key] = kwargs[key]

        torch.save(self.saving_buffer, self.args.logger.dir() + f'saving_buffer_{task_id}')

    def preprocess_task(self, **kwargs):
        # Add new embeddings for HAT
        self.net.append_embedddings()

        # Reset optimizer as there might be some leftover in optimizer
        if self.args.optimizer == 'sgd':
            self.optimizer = SGD(self.net.adapter_parameters(), lr=self.args.lr, momentum=self.args.momentum)
        elif self.args.optimizer == 'adam':
            raise NotImplementedError("HAT for Adam is not implemented")

        self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        
        # Prepare mask values for proper gradient update
        for n, p in self.net.named_parameters():
            p.grad = None
            if self.mask_back is not None:
                if n in self.mask_back.keys():
                    p.hat = self.mask_back[n]
                else:
                    p.hat = None
            else:
                p.hat = None

        # Prepare memory loader if memory data exist
        if self.args.use_buffer:
            if len(self.buffer_dataset.data) > 0:
                self.sampler = MySampler(
                    len(self.buffer_dataset),
                    len(kwargs['loader'].dataset),
                    self.args.batch_size,
                    kwargs['n_epochs'],
                )
                # We don't use minibatch. Use upsampling.
                self.buffer = DataLoader(self.buffer_dataset,
                                        batch_size=self.args.batch_size,
                                        sampler=self.sampler,
                                        num_workers=15,
                                        pin_memory=self.args.pin_memory)
                self.buffer_iter = iter(self.buffer)

    def end_task(self, **kwargs):
        test_loaders = kwargs['test_loaders']

        self.last_task_id += 1
        assert self.last_task_id + 1 == len(test_loaders)

        # Update masks for HAT
        self.p_mask = self.net.cum_mask(self.last_task_id, self.p_mask, self.args.smax)
        self.mask_back = self.net.freeze_mask(self.p_mask)

        # Update memory if used
        if self.args.use_buffer:
            self.buffer_dataset.update(kwargs['train_loader'].dataset)

            self.args.logger.print(Counter(self.buffer_dataset.targets))

            if os.path.exists(self.args.logger.dir() + f'/memory_{self.last_task_id}'):
                self.args.logger.print("Memory exists. Not saving memory...")
            else:
                self.args.logger.print("Saving memory...")
                torch.save([deepcopy(self.buffer_dataset.data),
                            deepcopy(self.buffer_dataset.targets)],
                           self.args.logger.dir() + f'/memory_{self.last_task_id}')
