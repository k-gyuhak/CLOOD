import os
import sys
from copy import deepcopy

import torch
import torch.nn as nn

from common import parse_args
from networks import create_network
from apprs import create_method
from datasets import create_dataset
from utils.utils import Logger, Criterion

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    args = parse_args()
    args.logger = Logger(args, args.folder)
    args.logger.now()

    args.device = device

    args.logger.print('\n\n',
                        os.uname()[1] + ':' + os.getcwd(),
                        'python', ' '.join(sys.argv),
                      '\n\n')

    args.logger.print('\n', args, '\n')
    
    args.criterion = Criterion(args)
    
    train_data, test_data = create_dataset(args)    
    args.net = create_network(args)
    model = create_method(args)
    
    if all(
        [
            args.test_id is None,
        ]
    ):   
        from row_pipeline import RowPipeline as Pipeline
        pipeline = Pipeline(args, train_data, test_data, model)
    
        args.logger.print("\nTraining starts\n")
        pipeline.train_all()
    
    elif args.test_id is not None:
        print("Testing")
        from base_pipeline import BasePipeline as Pipeline
        pipeline = Pipeline(args, train_data, test_data, model)
        
        for task_id in range(args.test_id + 1):
            pipeline.load_task_MD_stats(task_id)
            pipeline.preprocess_task(task_id)
            pipeline.load_train_step(task_id)
            pipeline.load_model_step(task_id)

        pipeline.test_auc(task_id, epoch=args.n_epochs)
        pipeline.print_results(task_id)