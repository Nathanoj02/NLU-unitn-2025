import math
import torch
import torch.nn as nn
from torch.optim import Optimizer


def train_loop(data, optimizer, criterion, model, clip=5):
    model.train()
    loss_array = []
    number_of_tokens = []
    
    for sample in data:
        optimizer.zero_grad()
        output = model(sample['source'])
        loss = criterion(output, sample['target'])
        loss_array.append(loss.item() * sample["number_tokens"])
        number_of_tokens.append(sample["number_tokens"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        optimizer.step()
        
    return sum(loss_array)/sum(number_of_tokens)


def eval_loop(data, eval_criterion, model):
    model.eval()
    loss_to_return = []
    loss_array = []
    number_of_tokens = []
    with torch.no_grad():
        for sample in data:
            output = model(sample['source'])
            loss = eval_criterion(output, sample['target'])
            loss_array.append(loss.item())
            number_of_tokens.append(sample["number_tokens"])
            
    ppl = math.exp(sum(loss_array) / sum(number_of_tokens))
    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    return ppl, loss_to_return

def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)


class NT_AvSGD(Optimizer):
    '''
    NT-AvSGD optimizer (Non-monotonically Triggered Averaged SGD)
    
    Args:
        params: Model parameters
        lr (float): Learning rate
        L (int): Logging interval (iterations per evaluation)
        n (int): Non-monotonic interval (consecutive non-improvements to trigger averaging)
    '''
    def __init__(self, params, lr, L, n = 5):
        defaults = dict(lr = lr, L = L, n = n)
        super().__init__(params, defaults)

        # State initialization
        self.k = 0          # Total iteration counter
        self.t = 0          # Validation evaluation counter
        self.T = 0          # Trigger iteration (0 = not triggered yet)
        self.logs = []      # Validation perplexity history
        self.best = float('inf')  # Best validation perplexity
        
        # Initialize weight averaging buffers
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['avg_sum'] = torch.zeros_like(p.data)
                state['avg_count'] = 0


    def step(self, closure=None):
        '''
        Performs a single optimization step
        '''
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        # Perform SGD update
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                p.data.add_(d_p, alpha=-lr)
        
        # Update averaging if triggered
        if self.T > 0:
            for group in self.param_groups:
                for p in group['params']:
                    state = self.state[p]
                    state['avg_sum'] += p.data
                    state['avg_count'] += 1
        
        self.k += 1
        return loss


    def update_val_loss(self, perplexity):
        '''
        Update validation loss and check triggering condition
        
        Args:
            perplexity (float): Current perplexity
        '''
        self.logs.append(perplexity)
        
        # Update best perplexity
        if perplexity < self.best:
            self.best = perplexity
        
        # Check triggering condition (after at least n+1 evaluations)
        if self.t >= self.defaults['n']:
            # Find minimum perplexity in last n evaluations (excluding current)
            min_prev = min(self.logs[-(self.defaults['n']+1):-1])
            
            # Trigger if current perplexity is worse than best in last n
            if perplexity > min_prev and self.T == 0:
                self.T = self.k  # Set trigger iteration
        
        self.t += 1


    def apply_average(self):
        '''
        Apply averaged weights to parameters
        '''
        if self.T == 0:
            return
        
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if state['avg_count'] > 0:
                    p.data.copy_(state['avg_sum'] / state['avg_count'])
            