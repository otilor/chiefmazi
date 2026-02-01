import torch
import torch.optim as optim


class SharedRMSprop(optim.RMSprop):
    """RMSprop with shared statistics across processes."""
    
    def __init__(self, params, lr=1e-4, alpha=0.99, eps=0.1):
        super().__init__(params, lr=lr, alpha=alpha, eps=eps)
        self._init_shared_state()

    def _init_shared_state(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1).share_memory_()
                state['square_avg'] = torch.zeros_like(p.data).share_memory_()
