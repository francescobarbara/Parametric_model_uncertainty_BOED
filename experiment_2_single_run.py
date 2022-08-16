import numpy as np
import torch
import numpy as np
import pyro
import pyro.distributions as dist
from torch.distributions.constraints import positive
from pyro.contrib.oed.eig import nmc_eig
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

def single_run(a, theta_real, number_experiments):
    '''
    Returns a list out where out[i]= [eigsi, ysi, lsi, historyi]
    That is, contains the full information for the run done using model_i
    Returns a list out where out[i]= [eigsi, ysi, lsi, historyi]
    That is, contains the full information for the run done using model_i.
    With full information we mean the expected information gains at each time step,
    the observed outcomes at each time step and the corresponding design chosen,
    plus a history object which contains a bunch of things such as the updated
    variational parameters of each distribution after each experiment is carried out,
    which we'll later use to compute RMSE'
    '''
    ''
    out = [] #out[i] = [eigsi, ysi, lsi, historyi]
    
    
    
    num_steps, start_lr, end_lr = 1000, 0.1, 0.001
    ys = torch.tensor([])
    ls = torch.tensor([])
    eigs = torch.tensor([])
    history = [(prior_mean, prior_sd)]
    pyro.clear_param_store()
    current_model1 = make_model1(prior_mean, prior_sd)
    
    for experiment in range(number_experiments):
        print("Round", experiment + 1)
    
        optimizer = pyro.optim.ExponentialLR({'optimizer': torch.optim.Adam,
                                              'optim_args': {'lr': start_lr},
                                              'gamma': (end_lr / start_lr) ** (1 / num_steps)})

        eig = nmc_eig(current_model1, l, observation_labels=["y"], target_labels=["theta"], N=100, M=10)

        best_eig = torch.max(eig).unsqueeze(0)
        eigs = torch.cat([eigs, best_eig], dim=0)
        
        best_l_index = torch.argmax(eig)
        best_l = l[best_l_index].clone()   
        best_l = best_l.unsqueeze(0)       

        y = generate_real(best_l)   
        ls = torch.cat([ls, best_l], dim=0)
        ys = torch.cat([ys, y], dim=0)
    
        conditioned_model1 = pyro.condition(current_model1, {"y": ys})

        svi = SVI(conditioned_model1,
                  guide1,
                  Adam({"lr": .005}),
                  loss=Trace_ELBO(),
                  num_samples=100)
        num_iters = 2000
        for i in range(num_iters):
            elbo = svi.step(ls)

        history.append((
                        pyro.param("posterior_mean").detach().clone(),
                        pyro.param("posterior_sd").detach().clone()))

        current_model1 = make_model1(pyro.param("posterior_mean").detach().clone(),
                                   pyro.param("posterior_sd").detach().clone())
        
    out.append([eigs, ls, ys, history])
    ############################################################################
    
    num_steps, start_lr, end_lr = 1000, 0.1, 0.001
    ys2 = torch.tensor([])
    ls2 = torch.tensor([])
    eigs2 = torch.tensor([])
    history2 = [(prior_mean, prior_sd, a_prior_mean, a_prior_sd)]
    pyro.clear_param_store()
    current_model2 = make_model2(prior_mean, prior_sd, a_prior_mean, a_prior_sd)
    
    for experiment in range(number_experiments):
        print("Round", experiment + 1)
    
        
        optimizer = pyro.optim.ExponentialLR({'optimizer': torch.optim.Adam,
                                              'optim_args': {'lr': start_lr},
                                              'gamma': (end_lr / start_lr) ** (1 / num_steps)})
        
        eig = nmc_eig(current_model2, l, observation_labels=["y"], target_labels=["theta"], N=100, M=10)
        
        best_eig = torch.max(eig).unsqueeze(0)
        eigs2 = torch.cat([eigs2, best_eig], dim=0)
        
        best_l_index = torch.argmax(eig)
        best_l = l[best_l_index].clone()   
        best_l = best_l.unsqueeze(0)
        
        y = generate_real(best_l)   
        
        ls2 = torch.cat([ls2, best_l], dim=0)
        ys2 = torch.cat([ys2, y], dim=0)
        
        conditioned_model2 = pyro.condition(current_model2, {"y": ys2})
        
        svi = SVI(conditioned_model2,
                  guide2,
                  Adam({"lr": .005}),
                  loss=Trace_ELBO(),
                  num_samples=100)
        num_iters = 2000
        for i in range(num_iters):
            elbo = svi.step(ls2)
        
        history2.append((
                        pyro.param("posterior_mean").detach().clone(),
                        pyro.param("posterior_sd").detach().clone(),
                        pyro.param("a_posterior_mean").detach().clone(),
                        pyro.param("a_posterior_sd").detach().clone()
                        ))
        
        current_model2 = make_model2(pyro.param("posterior_mean").detach().clone(),
                                   pyro.param("posterior_sd").detach().clone(),
                                   pyro.param("a_posterior_mean").detach().clone(),
                                   pyro.param("a_posterior_sd").detach().clone())
        
    out.append([eigs2, ls2, ys2, history2])
       
    num_steps, start_lr, end_lr = 1000, 0.1, 0.001
    ys3 = torch.tensor([])
    ls3 = torch.tensor([])
    eigs3 = torch.tensor([])
    history3 = [(prior_mean, prior_sd)]
    pyro.clear_param_store()
    current_model3 = make_model3(prior_mean, prior_sd)
    
    for experiment in range(number_experiments):
        print("Round", experiment + 1)

        optimizer = pyro.optim.ExponentialLR({'optimizer': torch.optim.Adam,
                                              'optim_args': {'lr': start_lr},
                                              'gamma': (end_lr / start_lr) ** (1 / num_steps)})
        
        eig = nmc_eig(current_model3, l, observation_labels=["y"], target_labels=["theta"], N=100, M=10)
        
        best_eig = torch.max(eig).unsqueeze(0)
        eigs3 = torch.cat([eigs3, best_eig], dim=0)
        
        best_l_index = torch.argmax(eig)
        best_l = l[best_l_index].clone()  
        best_l = best_l.unsqueeze(0)      
 
        y = generate_real(best_l)   
        ls3 = torch.cat([ls3, best_l], dim=0)
        ys3 = torch.cat([ys3, y], dim=0)
        
        conditioned_model3 = pyro.condition(current_model3, {"y": ys3})
        
        svi = SVI(conditioned_model3,
                  guide3,
                  Adam({"lr": .005}),
                  loss=Trace_ELBO(),
                  num_samples=100)
        num_iters = 2000
        for i in range(num_iters):
            elbo = svi.step(ls3)
        history3.append((
                        pyro.param("posterior_mean").detach().clone(),
                        pyro.param("posterior_sd").detach().clone(),
                        ))
        
        current_model3 = make_model3(pyro.param("posterior_mean").detach().clone(),
                                   pyro.param("posterior_sd").detach().clone()
                                   )
        
    out.append([eigs3, ls3, ys3, history3])

    num_steps, start_lr, end_lr = 1000, 0.1, 0.001
    ys2 = torch.tensor([])
    ls2 = torch.tensor([])
    eigs2 = torch.tensor([])
    history2 = [(prior_mean, prior_sd, a_prior_mean, a_prior_sd)]
    pyro.clear_param_store()
    current_model2 = make_model2(prior_mean, prior_sd, a_prior_mean, a_prior_sd)
    
    for experiment in range(number_experiments):
        print("Round", experiment + 1)
    
        
        optimizer = pyro.optim.ExponentialLR({'optimizer': torch.optim.Adam,
                                              'optim_args': {'lr': start_lr},
                                              'gamma': (end_lr / start_lr) ** (1 / num_steps)})
        
        eig = nmc_eig(current_model2, l, observation_labels=["y"], target_labels=["theta", "a"], N=100, M=10)
        
        best_eig = torch.max(eig).unsqueeze(0)
        eigs2 = torch.cat([eigs2, best_eig], dim=0)
        
        best_l_index = torch.argmax(eig)
        best_l = l[best_l_index].clone()   
        best_l = best_l.unsqueeze(0)       
        

        y = generate_real(best_l)    
        
        ls2 = torch.cat([ls2, best_l], dim=0)
        ys2 = torch.cat([ys2, y], dim=0)
    
        
        conditioned_model2 = pyro.condition(current_model2, {"y": ys2})
        
        svi = SVI(conditioned_model2,
                  guide2,
                  Adam({"lr": .005}),
                  loss=Trace_ELBO(),
                  num_samples=100)
        num_iters = 2000
        for i in range(num_iters):
            elbo = svi.step(ls2)
        
        history2.append((
                        pyro.param("posterior_mean").detach().clone(),
                        pyro.param("posterior_sd").detach().clone(),
                        pyro.param("a_posterior_mean").detach().clone(),
                        pyro.param("a_posterior_sd").detach().clone()
                        ))
        
        current_model2 = make_model2(pyro.param("posterior_mean").detach().clone(),
                                   pyro.param("posterior_sd").detach().clone(),
                                   pyro.param("a_posterior_mean").detach().clone(),
                                   pyro.param("a_posterior_sd").detach().clone())
        
    out.append([eigs2, ls2, ys2, history2])
    return out