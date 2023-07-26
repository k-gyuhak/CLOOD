from datasets.cont_data import get_data
from datasets.cont_data import StandardCL

def create_dataset(args):
    args.n_cls_task = int(args.n_cls // args.n_tasks)
    
    train_data, test_data = get_data(args)

    if args.task_type == 'standardCL_randomcls':
        train_data = StandardCL(train_data, args)
        test_data = StandardCL(test_data, args)
    return train_data, test_data