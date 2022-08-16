
import torch
import pyro
import pyro.distributions as dist
from torch.distributions.constraints import positive


def generate_real(l):
    '''This function will be used in our main function. There we have sampled 
        a shared true value of theta, 'theta_real', and whenever a BOED approach
        picks a specific design l, we output the corresponding experimental 
        outcome using this function'''    
    theta_prior_mean, theta_prior_sd = theta_real, torch.tensor([0.00001])
    
    with pyro.plate_stack("plate", l.shape[:-1]):       
        #a = pyro.sample("a", dist.LogNormal(alpha_prior_mean, alpha_prior_sd).to_event(1))
        
        theta = pyro.sample("theta", dist.Normal(theta_prior_mean, theta_prior_sd).to_event(1))
         
        denominator = (theta - l).pow(2).sum(axis = 1) + m 
        
        mu = a * denominator.pow(-1).unsqueeze(1) + b
        
        log_mu = torch.log(mu)
        y = pyro.sample("y", dist.Normal(log_mu, sd).to_event(1))     
        
    return y
