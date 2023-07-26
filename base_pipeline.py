from numpy.linalg import svd
from copy import deepcopy
from collections import Counter
from utils.utils import *
import torch
from tqdm import tqdm
import torchvision.transforms as transforms
from datetime import datetime

class BasePipeline:
    def __init__(self, args, train_data, test_data, model):
        self.args = args
        self.train_data = train_data
        self.test_data = test_data
        self.model = model

        self.preprocessed_task, self.received_data = -1, -1

        self.train_loaders, self.test_loaders = [], []

        self.trackers = {
            'cil_tracker': Tracker(args),
            'til_tracker': Tracker(args),
            'cal_cil_tracker': Tracker(args),
            'auc_tracker': AUCTracker(args),
            'auc_md_tracker': AUCTracker(args),
            'openworld_tracker': OWTracker(args),
            'openworld_md_tracker': OWTracker(args),
        }

        self.args.mean, self.args.cov, self.args.cov_inv = {}, {}, {}

        self.init_task = 0
        self.init_epoch = 0

        # Prepare all the data loaders for convenience
        for t in range(self.args.n_tasks): self.receive_data(t)

        if any([
            self.args.resume_id is not None,
            self.args.test_id is not None,
        ]):
            if self.args.load_path is None:
                self.args.load_path = self.args.logger.dir()

        if self.args.resume_id is not None:
            self.load_task(self.args.resume_id)

    def receive_all(self, task_id):
        for p_task_id in range(task_id + 1):
            self.receive_data(p_task_id)

    def receive_data(self, task_id):
        if len(self.train_loaders) <= task_id:
            self.args.logger.print(f"Received data of task {task_id}")
            if self.args.validation is None:
                t_train = self.train_data.make_dataset(task_id)
                t_test = self.test_data.make_dataset(task_id)
            else:
                t_train, t_test = self.train_data.make_dataset(task_id)

            self.train_loaders.append(make_loader(t_train, self.args, self.args.batch_size, train='train'))
            self.test_loaders.append(make_loader(t_test, self.args, self.args.test_batch_size, train='test'))

            self.current_train_loader = deepcopy(self.train_loaders[task_id])

    def load_train_step(self, task_id):
        checkpoint = custom_load(os.path.join(self.args.load_path, f'model_task_{task_id}'))

        # The old version only has model's state_dict. Check if it's old version or not
        if 'optimizer' not in checkpoint:
            custom_load_state_dict(self.args, self.model.net, checkpoint)
        else:
            custom_load_state_dict(self.args, self.model.net, checkpoint['state_dict'])

            self.init_task = checkpoint['task_id']
            self.init_epoch = checkpoint['epoch']

            if self.init_epoch >= self.args.n_epochs:
                self.init_task += 1
                self.init_epoch = 0

    def load_model_step(self, task_id=None):
        try:
            checkpoint = torch.load(os.path.join(self.args.load_path, f'saving_buffer_{task_id}'))
            for key, val in checkpoint.items():
                if hasattr(self.model, key):
                    self.args.logger.print(f"** {self.model.__class__.__name__}: Update {key} values **")
                    setattr(self.model, key, val)
                else:
                    self.args.logger.print(f"** WARNING: {self.model.__class__.__name__}: {key} values are not updated **")
        except FileNotFoundError:
            checkpoint = custom_load(os.path.join(self.args.load_path, f'memory_{task_id}'))
            self.model.buffer_dataset.update(self.current_train_loader.dataset)
            self.model.buffer_dataset.data = checkpoint[0]
            self.model.buffer_dataset.targets = checkpoint[1]

        if self.args.dataset in ['imgnet380', 'timgnet']:
            if hasattr(self.model, 'buffer_dataset'):
                temp = [_ for _ in self.model.buffer_dataset.data_list.keys()]
                if 'gkim87' in self.model.buffer_dataset.data_list[temp[0]][0]:
                    if 'gyuhak' in self.args.root:
                        for k in self.model.buffer_dataset.data_list.keys():
                            for i, val in enumerate(self.model.buffer_dataset.data_list[k]):
                                self.model.buffer_dataset.data_list[k][i] = val.replace('gkim87', 'gyuhak')
                elif 'gyuhak' in self.model.buffer_dataset.data_list[temp[0]][0]:
                    if 'gkim87' in self.args.root:
                        for k in self.model.buffer_dataset.data_list.keys():
                            for i, val in enumerate(self.model.buffer_dataset.data_list[k]):
                                self.model.buffer_dataset.data_list[k][i] = val.replace('gyuhak', 'gkim87')
                if 'gkim87' in self.model.buffer_dataset.data[0]:
                    if 'gyuhak' in self.args.root:
                        for i, val in enumerate(self.model.buffer_dataset.data):
                            self.model.buffer_dataset.data[i] = val.replace('gkim87', 'gyuhak')
                elif 'gyuhak' in self.model.buffer_dataset.data[0]:
                    if 'gkim87' in self.args.root:
                        for i, val in enumerate(self.model.buffer_dataset.data):
                            self.model.buffer_dataset.data[i] = val.replace('gyuhak', 'gkim87')

    def load_trackers(self, task_id=None):
        for k, v in self.trackers.items():
            try:
                v.mat = torch.load(os.path.join(self.args.logger.dir(), k))
                self.args.logger.print(f"Loaded {k}")
            except FileNotFoundError:
                pass

    def load_task(self, task_id):
        # self.receive_all(task_id)

        self.load_all_MD_stats(task_id)

        self.preprocess_all_tasks(task_id)

        self.load_train_step(task_id)
        self.load_model_step(task_id)
        self.load_trackers(task_id)

    def test_task(self, test_loader, **kwargs):
        """
            test_loader: test loader. This doesn't necessarily have to be same as task_id.
        """
        self.model.reset_eval()

        for data in test_loader:
            x, y = data['inputs'].to(self.args.device), data['targets'].to(self.args.device)

            self.model.evaluate(x, y, **kwargs)

        metrics = self.model.acc()

        return metrics

    def test_auc_new(self, task_id, metrics=None, **kwargs):
        """
            task_id: is the task id of provided results (the metrics in argument)
        """
        output_dict, md_dict, md_ow_dict, label_dict, pred_dict = {}, {}, {}, {}, {}

        try:
            epoch = kwargs['epoch']
        except KeyError:
            epoch = self.args.n_epochs - 1

        inputs_evaluate = {
            'task_id': task_id,
            'total_learned_task_id': self.preprocessed_task,
            'true_id': task_id,
            }

        if metrics is None: metrics = self.test_task(self.test_loaders[task_id], **inputs_evaluate)
        
        self.trackers['cil_tracker'].update(metrics['cil_acc'], task_id, task_id)
        self.trackers['til_tracker'].update(metrics['til_acc'], task_id, task_id)

        output_dict[task_id] = metrics['output_list']
        label_dict[task_id]  = metrics['label_list']
        pred_dict[task_id]   = metrics['pred_list']
        if self.args.compute_md: md_dict[task_id] = metrics['md_list']
        if self.args.ow_md: md_ow_dict[task_id] = metrics['md_ow_list']

        for task_out in range(self.args.n_tasks):
            if task_out != task_id:
                if self.args.validation is None:
                    t_test = self.test_data.make_dataset(task_out)
                else:
                    _, t_test = self.train_data.make_dataset(task_out)

                ood_loader = make_loader(t_test, self.args, self.args.test_batch_size, train='test')

                inputs_evaluate = {
                    'task_id': task_out,
                    'total_learned_task_id': task_id,
                    'true_id': task_out if task_out < task_id else None
                    }

                metrics = self.test_task(ood_loader, **inputs_evaluate)

                output_dict[task_out] = metrics['output_list']
                label_dict[task_out]  = metrics['label_list']
                pred_dict[task_out]   = metrics['pred_list']
                if self.args.compute_md: md_dict[task_out] = metrics['md_list']
                if self.args.ow_md: md_ow_dict[task_out] = metrics['md_ow_list']

            if task_out < task_id:
                self.trackers['cil_tracker'].update(metrics['cil_acc'], task_id, task_out)
                self.trackers['til_tracker'].update(metrics['til_acc'], task_id, task_out)

        if self.args.confusion:
            self.args.logger.print("Saving confusion matrix...")
            true_lab = np.concatenate([v for v in label_dict.values()])
            pred_lab = np.concatenate([v for v in pred_dict.values()])

            plot_confusion(true_lab, 
                pred_lab, 
                self.train_data.seen_names, 
                task_id, None, 
                logger=self.args.logger, 
                n_cls_task=self.args.n_cls_task,
                second_label_name=self.train_data.seen_names_coarse if self.args.dataset=='cifar100' else None)

        for data_id in range(self.preprocessed_task + 1):
            auc_arr = auc_output(output_dict,
                                 data_id,
                                 self.trackers['auc_tracker'],
                                 self.args.n_cls_task)

            if self.args.compute_md: auc_md_arr = auc(md_dict,
                                                     data_id,
                                                     self.trackers['auc_md_tracker'])

            if data_id == task_id:
                for task_out, val in enumerate(auc_arr):
                    if task_out != task_id:
                        self.args.logger.print("Epoch {}/{} | in/out: {}/{} | AUC: {:.2f}".format(epoch + 1, 
                            self.args.n_epochs, task_id, task_out, val), end=' ')

                        if self.args.compute_md:
                            self.args.logger.print("| MD AUC: {:.2f}".format(auc_md_arr[task_out]))
                        else:
                            self.args.logger.print()

        ow_arr = auc_output_ow(
            output_dict,
            task_id,
            self.trackers['openworld_tracker'],
            self.args.n_cls_task
        )

        if self.args.ow_md:
            ow_md_arr = auc_ow(
                md_ow_dict,
                task_id,
                self.trackers['openworld_md_tracker']
            )        

        for task_out, val in enumerate(ow_arr, task_id + 1):
            self.args.logger.print(
                "Epoch {}/{} | total in/out: {}/{} | AUC: {:.2f}".format(
                    epoch + 1,
                    self.args.n_epochs,
                    task_id,
                    task_out,
                    val
                )
            )

    def test_auc(self, task_id, metrics=None, **kwargs):
        """
            task_id: the task id the model will use (ind)
        """
        self.args.logger.print("This method is merged to test_auc_new()")
        self.test_auc_new(task_id, metrics, **kwargs)

    def save_trackers(self, **kwargs):
        for k, v in self.trackers.items():
            torch.save(v.mat, os.path.join(self.args.logger.dir(), k))

    def save_train_step(self, task_id, epoch, **kwargs):
        # Save anything relevant to training steps (e.g., epochs, task_id, optim, etc.)
        if hasattr(self.model, 'save'):
            if hasattr(self.model.net, 'adapter_state_dict'):
                state_dict = self.model.net.adapter_state_dict()
            elif hasattr(self.model.net, 'prompt_head_state_dict'):
                state_dict = self.model.net.prompt_head_state_dict()
            else:
                self.args.logger.print("******** Warning: Saving the entire network ********")
                state_dict = self.model.net.state_dict()

        # training specific
        checkpoint = {
            'state_dict': state_dict,
            'optimizer': self.model.optimizer.state_dict() if hasattr(self.model, 'optimizer') else None,
            'task_id': task_id,
            'epoch': epoch + 1,
        }

        torch.save(
            checkpoint,
            os.path.join(self.args.logger.dir(), f"model_task_{task_id}")
        )

    def save_model_step(self, task_id, **kwargs):
        # Save anything specific to current model
        self.model.save(task_id)

    def save_all(self, **kwargs):
        self.save_train_step(**kwargs)

        self.save_model_step(**kwargs)
        self.save_trackers(**kwargs)

    def load_task_MD_stats(self, task_id):
        try:
            cov = custom_load(
                os.path.join(self.args.load_path, f'{self.args.cov_task_name}_{task_id}.npy'),
                file_type='numpy'
            )
            self.args.cov[task_id] = cov
            self.args.cov_inv[task_id] = np.linalg.inv(0.8 * cov + 0.2 * np.eye(len(cov)))
            
            for y in range(task_id * self.args.n_cls_task, (task_id + 1) * self.args.n_cls_task):
                mean = custom_load(os.path.join(self.args.load_path,
                                                f'{self.args.mean_label_name}_{y}.npy'),
                                    file_type='numpy')
                self.args.mean[y] = mean
            self.args.logger.print("Means for classes:", self.args.mean.keys())
            self.args.logger.print("Covs for classes:", self.args.cov.keys())
        except FileNotFoundError:
            self.args.logger.print(f"*** No MD for Task {task_id}***")

    def load_all_MD_stats(self, task_id):
        for p_task_id in range(task_id + 1):
            self.load_task_MD_stats(p_task_id)

    def preprocess_all_tasks(self, task_id):
        for p_task_id in range(task_id + 1):
            self.preprocess_task(p_task_id)

    def preprocess_task(self, task_id):
        self.current_train_loader = self.train_loaders[task_id]

        if self.preprocessed_task != task_id:
            self.args.logger.print(f"Preprocessing task {task_id}")
            inputs_preprocess = {
                'loader': self.current_train_loader,
                'n_epochs': self.args.n_epochs
                }

            if hasattr(self.model, 'preprocess_task'):
                self.model.preprocess_task(**inputs_preprocess)
                self.args.logger.print(Counter(self.current_train_loader.dataset.targets))

            self.preprocessed_task = task_id

    def train_task(self, task_id):
        self.preprocess_task(task_id)

        for epoch in range(self.init_epoch, self.args.n_epochs):
            inputs_observe = {
                'task_id': task_id,
                'B': len(self.current_train_loader),
            }

            task_loss_list, cum_acc_list = [], []
            self.model.reset_eval()

            for b, data in tqdm(enumerate(self.current_train_loader)):
                x = data['inputs'].to(self.args.device)
                y = data['targets'].to(self.args.device)
                orig = data['orig'].to(self.args.device)
                names = data['names']

                inputs_observe['b'] = b
                inputs_observe['indices'] = data['indices']
                inputs_observe['transform'] = transforms.Compose(
                    [
                        transforms.ToPILImage(),
                        self.current_train_loader.dataset.transform
                    ]
                )

                loss = self.model.observe(x, y, names, orig, **inputs_observe)

                task_loss_list.append(loss)
                cum_acc_list.append(self.model.correct / self.model.total * 100)

            self.model.end_epoch()
            self.args.logger.print(
                "Epoch {}/{} | Loss: {:.4f} | Acc: {:.2f}".format(
                    epoch + 1,
                    self.args.n_epochs,
                    self.model.total_loss / len(self.current_train_loader),
                    self.model.correct / self.model.total * 100
                )
            )

            if any([
                (epoch + 1) % self.args.eval_every == 0,
                (epoch + 1) == self.args.n_epochs
            ]):
                inputs_evaluate = {
                    'task_id': task_id,
                    'total_learned_task_id': task_id,
                }
                metrics = self.test_task(self.test_loaders[task_id], **inputs_evaluate)
                
                self.args.logger.print(
                    "Task {}, Epoch {}/{}, Total Loss: {:.4f}, CIL Acc: {:.2f}, TIL Acc: {:.2f}".format(
                        task_id,
                        epoch + 1,
                        self.args.n_epochs,
                        np.mean(task_loss_list),
                        metrics['cil_acc'],
                        metrics['til_acc']
                    )
                )

                inputs_save = {
                    'task_id': task_id,
                    'epoch': epoch,
                }
                self.save_all(**inputs_save)
    
    def end_task(self, task_id):
        if self.args.compute_md:
            self.model.compute_stats(task_id, self.train_loaders[task_id])
        
        self.test_auc_new(task_id, None, epoch=self.args.n_epochs)

        inputs_save = {
            'task_id': task_id,
            'epoch': self.args.n_epochs,
        }
        self.save_all(**inputs_save)
        
        self.args.logger.print("End task...")
        inputs_end_task = {
            'test_loaders': self.test_loaders[:task_id + 1],
            'train_loader': self.train_loaders[task_id],
            'task_id': task_id
        }
        if hasattr(self.model, 'end_task'):
            self.model.end_task(**inputs_end_task)

        self.print_result(task_id)
        self.save_trackers()

    def train_all(self):
        for task_id in range(self.init_task, self.args.n_tasks):
            self.train_task(task_id)
            self.end_task(task_id)

    def test_all_all(self, task_id):
        """
            Test all from task 0 to task_id. Task task model must be saved.
        """
        for p_task_id in range(0, task_id + 1):
            self.load_task(p_task_id)
            self.print_result(p_task_id)

    def print_result(self, task_id):
        self.args.logger.print("######################")

        self.args.logger.print()
        if self.args.compute_auc:
            self.args.logger.print("AUC result")
            self.trackers['auc_tracker'].print_result(task_id, type='acc')
            self.args.logger.print("Open World result (output)")
            self.trackers['openworld_tracker'].print_result(task_id, type='acc')
        if self.args.ow_md:
            self.args.logger.print("Open World result (MD)")
            self.trackers['openworld_md_tracker'].print_result(task_id, type='acc')
        if self.args.compute_md:
            self.args.logger.print("MD AUC result")
            self.trackers['auc_md_tracker'].print_result(task_id, type='acc')
        self.args.logger.print("CIL result")
        self.trackers['cil_tracker'].print_result(task_id, type='acc')
        self.trackers['cil_tracker'].print_result(task_id, type='forget')
        self.args.logger.print("TIL result")
        self.trackers['til_tracker'].print_result(task_id, type='acc')
        self.trackers['til_tracker'].print_result(task_id, type='forget')
        self.args.logger.print()
        self.args.logger.now()