from argparse import ArgumentParser

def parse_args():
    # Admin
    parser = ArgumentParser()
    parser.add_argument(
        '--root', type=str, default='/home/gyuhak/data',
        help="Path to the data"
    )
    parser.add_argument(
        '--folder', type=str, default=None,
        help='directory NAME. e.g. save under ./logs/NAME'
    )
    parser.add_argument(
        '--print_filename', type=str, default=None,
        help="Write all the print on the provided filename under the folder specified in the argument 'folder'. If None, write on 'result.txt' file"
    )
    parser.add_argument(
        '--task_type', type=str, default='standardCL_randomcls', choices=['standardCL_randomcls'],
        help="Learning scenario"
    )
    parser.add_argument(
        '--eval_every', type=int, default=5,
        help="Evaluate the model at every xx-epoch increment"
    )
    parser.add_argument(
        '--exe', action='store_true',
        help="If true, execute the code using 'exe_n_samples' samples per class. Use it for checking sanity of the code"
    )
    parser.add_argument(
        '--exe_n_samples', type=int, default=20,
        help="The number of samples used per class. Only useful when the argument 'exe' is True"
    )
    parser.add_argument(
        '--validation', type=float, default=None,
        help="Propertion of dataset used for validation. For instance, if set 0.9, 90\% of the training data is used for training and the remaining 10\% is used for validation"
    )
    parser.add_argument('--seed', type=int, default=0)
    
    # CL setting
    parser.add_argument(
        '--architecture', type=str, default='deit_small_patch16_224',
        help="Network architecture e.g., resnet, deit, vit, alexnet, etc."
    )
    parser.add_argument(
        '--transformer', default=None, type=str, choices=['adapter', 'adapter_hat'],
        help="Transformer with or without adapter and HAT"
    )
    parser.add_argument(
        '--n_pre_cls', default=None, type=int,
        help="Number of classes used for pre-training the transformer network. If None, load the checkpoint pre-trained with 611 classes of ImageNet"
    )
    parser.add_argument(
        '--method', type=str, default=None, choices=['ROW', 'HAT'],
        help="CL method"
    )
    parser.add_argument(
        '--dataset', type=str, default='cifar100', choices=['mnist', 'svhn', 'cifar100', 'cifar10', 'timgnet', 'imgnet380'],
    )
    parser.add_argument(
        '--class_order', type=int, default=0, choices=[0, 1, 2, 3, 4],
        help="Class order"
    )
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['adam', 'sgd'])
    parser.add_argument('--scheduler', type=str, default=None, choices=['multistep', 'cosine', 'steplr'])
    parser.add_argument(
        '--steps', type=int, nargs='*', default=[80, 140],
        help="The steps for multistep lr scheduler"
    )
    parser.add_argument(
        '--steps_gamma', type=float, default=0.1,
        help="The gamma value for multistep lr scheduler"
    )
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument(
        '--momentum', type=float, default=0.9, help="Momentum value for sgd"
    )
    parser.add_argument(
        '--n_epochs', type=int, default=1,
        help="Number of epochs for the main training"
    )
    parser.add_argument(
        '--n_tasks', type=int, default=5,
        help="Number of tasks for continual learning"
    )
    parser.add_argument(
        '--loss_f', type=str, default='ce', choices=['ce', 'bce', 'nll'],
        help="Loss function for the main training"
    )
    parser.add_argument(
        '--test_id', type=int, default=None,
        help="If provided, test the model for the provided task id. Task id starts from 0"
    )
    
    # Adapter
    parser.add_argument(
        '--adapter_latent', type=int, default=64, help="Size of the adapter"
    )
    
    # DataLoader
    parser.add_argument('--pin_memory', action='store_false')
    parser.add_argument('--num_workers', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=512)
    
    # Replay
    parser.add_argument(
        '--use_buffer', action='store_true',
        help="If true, use memory buffer"
    )
    parser.add_argument(
        '--buffer_size', type=int, default=200,
        help="Size of the memory buffer. By default, save equal number of samples per class"
    )
    
    # For HAT
    parser.add_argument('--smax', type=float, default=500)
    parser.add_argument(
        '--lamb0', type=float, default=0.75,
        help="The weight on HAT regularization at task=0"
    )
    parser.add_argument(
        '--lamb1', type=float, default=0.75,
        help="The weight on HAT regularization at task > 0"
    )
    parser.add_argument('--thres_cosh', type=float, default=50)
    parser.add_argument('--thres_emb', type=float, default=6)
    
    # ROW
    parser.add_argument(
        '--finetune_wp_epochs', default=5, type=int,
        help="Finetune the WP head"
    )
    parser.add_argument(
        '--finetune_ood_epochs', default=10, type=int,
        help="Finetune the TP heads for each task"
    )
    parser.add_argument(
        '--T', type=float, default=5,
        help="Temperature scaling for TP and WP probabilities. This improves the CIL accuracy from each task as the OOD performance improves. See WPTP (https://arxiv.org/pdf/2211.02633v1.pdf)"
    )
    parser.add_argument(
        '--wp_head_name', default='wp_head', type=str,
        help="Save the WP head by the provided name"
    )
    parser.add_argument(
        '--tp_head_name', default='tp_head', type=str,
        help="Save the TP heads by the provided file name"
    )
    parser.add_argument(
        '--batch_size_finetune', type=int, default=32,
        help="Batch size for fine-tuning WP and OOD heads"
    )
    
    # Multi-head to CIL
    parser.add_argument(
        '--task_inference', type=str, default=None, choices=['entropy'],
        help="Specify the task-id inference method. If None, follow the method in WPTP (https://arxiv.org/pdf/2211.02633v1.pdf)"
    )
    
    # Features
    parser.add_argument(
        '--compute_md', action='store_true',
        help='If true, compute the statistics of faetures for mahalanobis distance'
    )
    parser.add_argument(
        '--use_md', action='store_true',
        help="If true, use Mahalanobis distance for CIL prediction"
    )
    parser.add_argument(
        '--mean_label_name', type=str, default='mean_label',
        help="File name for saving the mean vector of features of a class"
    )
    parser.add_argument(
        '--cov_task_name', type=str, default='cov_task',
        help="File name for saving the covariance matrix of features of the task"
    )
    parser.add_argument(
        '--ow_md', action='store_true',
        help="If true, compute the open-world AUC using the Mahalanobis distance at feature level"
    )

    parser.add_argument('--init_task', type=int, default=0)
    parser.add_argument('--init_epoch', type=int, default=0, help='initial epoch. Epoch starts from init_epoch and finishes at n_epochs-1')
    
    # Misc
    parser.add_argument(
        '--compute_auc', action='store_true',
        help="If true, compute the AUC for each task as (https://arxiv.org/pdf/2211.02633.pdf) and open-world AUC as (https://arxiv.org/pdf/2208.09734.pdf)"
    )
    parser.add_argument('--confusion', action='store_true')
    parser.add_argument('--tsne', action='store_true')
    
    parser.add_argument('--resume_id', type=int, default=None, help='resume id. If provided, training begins when task_id == resume_id')
    parser.add_argument('--resume', type=str, default=None, help='resume path')
    parser.add_argument('--load_path', type=str, default=None)

    args = parser.parse_args()
    if args.dataset == 'mnist':
        args.n_cls = 10
    elif args.dataset == 'svhn':
        args.n_cls = 10
    elif args.dataset == 'cifar10':
        args.n_cls = 10
    elif args.dataset == 'cifar100':
        args.n_cls = 100
    elif args.dataset == 'timgnet':
        args.n_cls = 200
    elif args.dataset == 'imgnet380':
        args.n_cls = 380
    else:
        raise NotImplementedError()

    return args