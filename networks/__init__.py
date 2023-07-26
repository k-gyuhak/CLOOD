import importlib
import torch
from utils.utils import load_pretrain

device = "cuda" if torch.cuda.is_available() else "cpu"

def create_network(args):
    if 'deit' or 'vit' in args.architecture:
        mod = importlib.import_module('networks.' + 'my_vision_transformer')
        transformer = getattr(mod, args.architecture)
        
        if 'ROW' in args.method and args.use_buffer:
            num_classes = args.n_cls_task + 1
        elif 'ROW' in args.method and not args.use_buffer:
            num_classes = args.n_cls_task
            
        net = transformer(pretrained=False, num_classes=num_classes, latent=args.adapter_latent, transformer=args.transformer).to(device)
        args.in_dim = net.num_features
        net = load_pretrain(args, net, args.n_pre_cls)
        
    else:
        raise NotImplementedError()   

    return net