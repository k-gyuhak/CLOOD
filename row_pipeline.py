from copy import deepcopy
from collections import Counter
from utils.utils import *
import torch
from tqdm import tqdm

from base_pipeline import BasePipeline

class RowPipeline(BasePipeline):
    def __init__(self, args, train_data, test_data, model):
        super().__init__(args, train_data, test_data, model)
        self.model.net.tp_head = nn.ModuleList()
    
    def train_all(self):
        for task_id in range(self.init_task, self.args.n_tasks):
            self.args.use_two_heads = False
            self.train_task(task_id)
            
            if task_id > 0:
                self.finetune_wp(task_id)
                
                self.args.use_two_heads = True
                self.finetune_all_ood(task_id)
            
            self.end_task(task_id)
                
    def finetune_all_ood(self, task_id):
        sample_per_cls = Counter(self.model.buffer_dataset.targets)
        self.args.logger.print("Number of samples per class: ", sample_per_cls)
        sample_per_cls = sample_per_cls[0]

        current_loader_copy = deepcopy(self.finetune_train_loader)
        buffer_copy = deepcopy(self.model.buffer_dataset)
        if isinstance(self.finetune_train_loader.dataset, Subset):
            indices = np.concatenate([self.finetune_train_loader.dataset.indices, self.model.buffer_dataset.indices])
            data = [self.finetune_train_loader.dataset.dataset.samples[i][0] for i in indices]
            data = np.concatenate([data, self.model.buffer_dataset.data])
        else:
            data = np.concatenate([self.finetune_train_loader.dataset.data, self.model.buffer_dataset.data])
        targets = np.concatenate([self.finetune_train_loader.dataset.targets, self.model.buffer_dataset.targets])
        
        for p_task_id in range(task_id + 1):
            ind = set(np.arange(p_task_id * self.args.n_cls_task, (p_task_id + 1) * self.args.n_cls_task))
            
            ind_list, ood_list = [], []
            for y in range((task_id + 1) * self.args.n_cls_task):
                idx = np.where(targets == y)[0]
                np.random.shuffle(idx)
                idx = idx[:sample_per_cls]
                
                if y in ind:
                    ind_list.append(idx)
                else:
                    ood_list.append(idx)
            
            self.finetune_train_loader.dataset.targets = targets[np.concatenate(ind_list)]
            if isinstance(self.finetune_train_loader.dataset, Subset):
                self.finetune_train_loader.dataset.indices = indices[np.concatenate(ind_list)]
            else:
                self.finetune_train_loader.dataset.data = data[np.concatenate(ind_list)]
            self.model.buffer_dataset.data = data[np.concatenate(ood_list)]
            self.model.buffer_dataset.targets = targets[np.concatenate(ood_list)]

            self.finetune_ood(p_task_id, task_id)     
        
        self.finetune_train_loader = current_loader_copy
        self.model.buffer_dataset = buffer_copy
            
        checkpoint = self.model.net.tp_head.state_dict()
        torch.save(
            checkpoint,
            os.path.join(self.args.logger.dir(), f'{self.args.tp_head_name}_{task_id}')
        )

    def finetune_ood(self, task_id, last_task_id=None):
        self.args.logger.print(f"Fine-tuning an OOD head task={task_id}")
        self.model.preprocess_finetune_ood(
            task_id=task_id,
            loader=self.finetune_train_loader,
            n_epochs=self.args.finetune_ood_epochs
        )

        for epoch in range(self.args.finetune_ood_epochs):
            task_loss_list = []
            for b, data in tqdm(enumerate(self.finetune_train_loader)):
                x = data['inputs'].to(self.args.device)
                y = data['targets'].to(self.args.device)

                loss = self.model.observe_ood(x, y, task_id=task_id)
                task_loss_list.append(loss)

            if (epoch + 1) == self.args.finetune_ood_epochs:
                inputs_evaluate = {
                    'task_id': task_id,
                    'total_learned_task_id': task_id,
                    }
                metrics = self.test_task(self.test_loaders[task_id], **inputs_evaluate)
                
                self.args.logger.print(
                    "Task {}, Epoch {}/{}, Total Loss: {:.4f}, CIL Acc: {:.2f}, TIL Acc: {:.2f}".format(
                        task_id,
                        epoch + 1,
                        self.args.finetune_ood_epochs,
                        np.mean(task_loss_list),
                        metrics['cil_acc'], metrics['til_acc']
                    )
                )

    def finetune_wp(self, task_id):
        self.args.logger.print("Fine-tuning WP head")

        # self.finetune_train_loader = deepcopy(self.train_loaders[task_id])
        if self.args.validation is None:
            t_train = self.train_data.make_dataset(task_id)
        else:
            t_train, _ = self.train_data.make_dataset(task_id)
        self.finetune_train_loader = make_loader(t_train, self.args, self.args.batch_size_finetune, train='train')


        self.model.preprocess_finetune_wp(
            task_id=task_id,
            loader=self.finetune_train_loader
        )

        for epoch in range(self.args.finetune_wp_epochs):
            task_loss_list = []
            for b, data in tqdm(enumerate(self.finetune_train_loader)):
                x = data['inputs'].to(self.args.device)
                y = data['targets'].to(self.args.device)

                loss = self.model.observe_wp(x, y, task_id=task_id)
                task_loss_list.append(loss)

            if (epoch + 1) == self.args.finetune_wp_epochs:
                inputs_evaluate = {
                    'task_id': task_id,
                    'total_learned_task_id': task_id,
                }

                metrics = self.test_task(self.test_loaders[task_id], **inputs_evaluate)
                
                self.args.logger.print(
                    "Task {}, Epoch {}/{}, Total Loss: {:.4f}, CIL Acc: {:.2f}, TIL Acc: {:.2f}".format(
                        task_id,
                        epoch + 1,
                        self.args.finetune_wp_epochs,
                        np.mean(task_loss_list),
                        metrics['cil_acc'],
                        metrics['til_acc']
                    )
                )

        checkpoint = self.model.net.head.state_dict()
        torch.save(
            checkpoint,
            os.path.join(self.args.logger.dir(), f'{self.args.wp_head_name}_{task_id}')
        )
