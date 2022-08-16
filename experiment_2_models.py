import torch
import numpy as np
import pyro
import pyro.distributions as dist
from torch.distributions.constraints import positive

def model1(l):
    #Model 1 represents the fully specified model
    #Given a tensor l, it returns l.shape[:-1] independent experimental 
    #outcomes y where each y_i is associated to a choice of xi_i (in this case 
    #elements of l) and a value of theta sampled from the prior
    
    with pyro.plate_stack("plate", l.shape[:-1]):       

        theta = pyro.sample("theta", dist.Normal(theta_prior_mean, theta_prior_sd).to_event(1))
        denominator = (theta - l).pow(2).sum(axis = 1) + m 
        mu = a * denominator.pow(-1).unsqueeze(1) + b        
        log_mu = torch.log(mu)
        y = pyro.sample("y", dist.Normal(log_mu, sd).to_event(1))
 
        return y

def model2(l):
    #Model 2 can be used for both PMU and for maximisation of the joint EIG.
    #Given a tensor l, it returns l.shape[:-1] independent experimental 
    #outcomes y where each y_i is associated to a choice of xi_i (in this case 
    #elements of l), a value of theta and a value of alpha sampled from the
    #respective priors
    with pyro.plate_stack("plate", l.shape[:-1]):       
        a = pyro.sample("a", dist.LogNormal(alpha_prior_mean, alpha_prior_sd).to_event(1))      
        theta = pyro.sample("theta", dist.Normal(theta_prior_mean, theta_prior_sd).to_event(1))       
        denominator = (theta - l).pow(2).sum(axis = 1) + m    
        mu = a * denominator.pow(-1).unsqueeze(1) + b
        log_mu = torch.log(mu)
        y = pyro.sample("y", dist.Normal(log_mu, sd).to_event(1))
 
        return y
    
def model3(l):
    #Model 3 represents the misspecified model.
    #Given a tensor l, it returns l.shape[:-1] independent experimental 
    #outcomes y where each y_i is associated to a choice of xi_i (in this case 
    #elements of l) a value of theta sampled from the prior and a misspecified
    #value of alpha
    with pyro.plate_stack("plate", l.shape[:-1]):       
        theta = pyro.sample("theta", dist.Normal(theta_prior_mean, theta_prior_sd).to_event(1))
        denominator = (theta - l).pow(2).sum(axis = 1) + m 
        mu = a_fixed * denominator.pow(-1).unsqueeze(1) + b
        log_mu = torch.log(mu)
        y = pyro.sample("y", dist.Normal(log_mu, sd).to_event(1))
 
        return y