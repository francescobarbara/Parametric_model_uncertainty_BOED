import torch
import pyro
import pyro.distributions as dist
from torch.distributions.constraints import positive

'''Guides are used to define the Variational familes on which we will be
    carrying out SVI. Each of the guides below is matched to a corresponding 
    model from above. We use Gaussians for variational inference, with initial 
    parameters initialised as the prior parameters.'''
            
def guide1(l):
    # The guide is initialised at the prior
    posterior_mean = pyro.param("posterior_mean", prior_mean.clone())
    posterior_sd = pyro.param("posterior_sd", prior_sd.clone(), constraint=positive)
    
    with pyro.plate_stack("plate", l.shape[:-1]):  
        pyro.sample("theta", dist.Normal(posterior_mean, posterior_sd).to_event(1))
        
def guide2(l):
    # The guide is initialised at the prior
    posterior_mean = pyro.param("posterior_mean", prior_mean.clone())
    posterior_sd = pyro.param("posterior_sd", prior_sd.clone(), constraint=positive)
    
    a_posterior_mean = pyro.param("a_posterior_mean", a_prior_mean.clone())
    a_posterior_sd = pyro.param("a_posterior_sd", a_prior_sd.clone(), constraint=positive)
    
    with pyro.plate_stack("plate", l.shape[:-1]):  
        pyro.sample("theta", dist.Normal(posterior_mean, posterior_sd).to_event(1))
        pyro.sample("a", dist.Normal(a_posterior_mean, a_posterior_sd).to_event(1))
        
def guide3(l):
    # The guide is initialised at the prior
    posterior_mean = pyro.param("posterior_mean", prior_mean.clone())
    posterior_sd = pyro.param("posterior_sd", prior_sd.clone(), constraint=positive)
    
    with pyro.plate_stack("plate", l.shape[:-1]):  
        pyro.sample("theta", dist.Normal(posterior_mean, posterior_sd).to_event(1))