import torch
from torch.optim.optimizer import required
from torch.optim import SGD

class SGD_hat(SGD):
    def __init__(self, params, lr=required, momentum=0, dampening=0, 
                 weight_decay=0, nesterov=False):
        super(SGD_hat, self).__init__(params, lr, momentum, dampening,
                                      weight_decay, nesterov)

    @torch.no_grad()
    def step(self, closure=None, hat=False):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf
                if hat:
                    if p.hat is not None:
                        d_p = d_p * p.hat
                p.add_(d_p, alpha=-group['lr'])

        return loss