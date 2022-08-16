import torch

def create_designs(x_lower=-1, x_upper=1, x_step=1, y_lower=-1, y_upper=1, y_step=1):
    '''Creates the experimental designs to evaluate'''
    x, y = torch.meshgrid([torch.arange(x_lower, x_upper, x_step), torch.arange(y_lower, y_upper, y_step)])
    cols = torch.prod(torch.tensor(x.shape))
    x = x.reshape([cols, 1])
    
    
    cols2 = torch.prod(torch.tensor(y.shape))
    y = y.reshape([cols2,1])
    
    
    z = torch.cat([x, y], axis = 1)
    return z