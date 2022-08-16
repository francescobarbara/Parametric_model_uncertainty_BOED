import numpy as np
import torch
import pyro
import pyro.distributions as dist
from torch.distributions.constraints import positive
from pyro.contrib.oed.eig import nmc_eig
from pyro.contrib.oed.eig import posterior_eig
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

def multiple_runs(number_runs, number_experiments):
    
    '''It performs 'number_runs' experimental runs, each of length 
        'number_experiments' and returns information on them as a list of history
        objects earlier defined in 'single_run.py', it also returns the true hidden
        values of theta and alpha for each run'''
    
    full_histories = []
    full_true_values = []
    for i in range(number_runs):
        print('number_run', i)
        global a
        a = pyro.sample('a', dist.Normal(a_prior_mean, a_prior_sd).to_event(1))
        global theta_real
        theta_real = pyro.sample("theta",  dist.Normal(prior_mean, prior_sd).to_event(1))
        
        true_values = [theta_real.clone(), a.clone()]
        full_true_values.append(true_values)
        
        out = single_run(a, theta_real, number_experiments)
        full_histories.append(out)
    return (full_histories, full_true_values)