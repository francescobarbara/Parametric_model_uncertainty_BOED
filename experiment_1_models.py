import torch
import pyro
import pyro.distributions as dist

def model1(l):
    #Model 1 represents the fully specified model
    #Given a tensor l, it returns l.shape[:-1] independent experimental 
    #outcomes y where each y_i is associated to a choice of xi_i (in this case 
    #elements of l) and a value of theta sampled from the prior
    
    with pyro.plate_stack("plate", l.shape[:-1]):       
    
        theta = pyro.sample("theta",  dist.Normal(prior_mean, prior_sd).to_event(1))
        distance = l - theta
        inner_quantity = a - c* (distance.pow(2))
        probability = (torch.tensor(1.) + torch.exp(inner_quantity)).pow(-1)
        y = pyro.sample("y", dist.Binomial(10, probability).to_event(1))
       
        return y

def model2(l):
    #Model 2 can be used for both PMU and for maximisation of the joint EIG.
    #Given a tensor l, it returns l.shape[:-1] independent experimental 
    #outcomes y where each y_i is associated to a choice of xi_i (in this case 
    #elements of l), a value of theta and a value of alpha sampled from the
    #respective priors
    
    with pyro.plate_stack("plate", l.shape[:-1]):       
    
        theta = pyro.sample("theta",  dist.Normal(prior_mean, prior_sd).to_event(1))
        a = pyro.sample("a", dist.Normal(a_prior_mean, a_prior_sd))
        distance = l - theta
        inner_quantity = a - c* (distance.pow(2))
        probability = (torch.tensor(1.) + torch.exp(inner_quantity)).pow(-1)
        y = pyro.sample("y", dist.Binomial(10, probability).to_event(1))
 
        return y
    
def model3(l):
    #Model 3 represents the misspecified model.
    #Given a tensor l, it returns l.shape[:-1] independent experimental 
    #outcomes y where each y_i is associated to a choice of xi_i (in this case 
    #elements of l) a value of theta sampled from the prior and a misspecified
    #value of alpha
    
    with pyro.plate_stack("plate", l.shape[:-1]):       
    
        theta = pyro.sample("theta",  dist.Normal(prior_mean, prior_sd).to_event(1))
        distance = l - theta
        inner_quantity = a_fixed - c* (distance.pow(2))
        probability = (torch.tensor(1.) + torch.exp(inner_quantity)).pow(-1)
        y = pyro.sample("y", dist.Binomial(10, probability).to_event(1))
     
        return y