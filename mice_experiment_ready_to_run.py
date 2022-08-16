
import numpy as np
import torch
import numpy as np
import pyro
import pyro.distributions as dist
from torch.distributions.constraints import positive
from pyro.contrib.oed.eig import marginal_eig
from pyro.contrib.oed.eig import nmc_eig
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

'''Defining initial parameters and distributions of interest'''
prior_mean = torch.tensor([3.5])
prior_sd = torch.tensor([0.75])

l = torch.arange(2.,6.,0.1).unsqueeze(1)
a = torch.tensor([3.1])
c = torch.tensor([2.9])


    
def model(l):
    #Given a tensor l, it returns l.shape[:-1] independent experimental 
    #outcomes y where each y_i is associated to a choice of xi_i (in this case 
    #elements of l) and a value of theta sampled from the prior
    with pyro.plate_stack("plate", l.shape[:-1]):       
    
        theta = pyro.sample("theta",  dist.Normal(prior_mean, prior_sd).to_event(1))
        #print(theta)  
        distance = l - theta
        inner_quantity = a - c* (distance.pow(2))
        probability = (torch.tensor(1.) + torch.exp(inner_quantity)).pow(-1)
        #print(probability)
        y = pyro.sample("y", dist.Binomial(10, probability).to_event(1))
 
       
        return y

#############################################################################
            
def guide(l):
    '''Variational family used for inference of the posterior after each 
        experimental outcome is observed'''
        
    posterior_mean = pyro.param("posterior_mean", prior_mean.clone())
    posterior_sd = pyro.param("posterior_sd", prior_sd.clone(), constraint=positive)
    
    with pyro.plate_stack("plate", l.shape[:-1]):  
        pyro.sample("theta", dist.Normal(posterior_mean, posterior_sd).to_event(1))


#Sanity check, Variational Inference working as expected!
#4 experiments chosen ahead time, sub-optimal strategy
l_data = torch.tensor([2., 3., 4., 5.])
y_data = generate_real(l_data)



conditioned_model = pyro.condition(model, {"y": y_data})  
svi = SVI(conditioned_model,
          guide,
          Adam({"lr": .001}),
          loss=Trace_ELBO(),
          num_samples=100)
pyro.clear_param_store()
num_iters = 5000
for i in range(num_iters):
    elbo = svi.step(l_data)        
    if i % 500 == 0:
        print("Neg ELBO:", elbo)
        
print(pyro.param('posterior_mean'), pyro.param('posterior_sd'))
# posterior_mean = 4.47
# posterior_sd = 0.21
##############################################################################       


def generate_real(l):
    'Generates real ys given a design l chosen'  
    theta_prior_mean, theta_prior_sd = torch.tensor([4.27]), torch.tensor([0.00001])
    
          
    with pyro.plate_stack("plate", l.shape[:-1]): 
        theta = pyro.sample("theta", dist.Normal(theta_prior_mean, theta_prior_sd).to_event(1))
        
        distance = l - theta
        inner_quantity = a - c* (distance.pow(2))
        probability = (torch.tensor(1.) + torch.exp(inner_quantity)).pow(-1)
        #print(probability)
        y = pyro.sample("y", dist.Binomial(10, probability).to_event(1))
        
        
    return y

##############################################################################
''' 'make_model' is a wrapper around the earlier defined models that allow
    us to easily create models with updated priors during a full run of a 
    sequential experiment'''
    
def make_model(prior_mean, prior_sd):
    def model(l):
    
        with pyro.plate_stack("plate", l.shape[:-1]):       
        
            theta = pyro.sample("theta",  dist.Normal(prior_mean, prior_sd).to_event(1))
            #print(theta)  
            distance = l - theta
            inner_quantity = a - c* (distance.pow(2))
            probability = (torch.tensor(1.) + torch.exp(inner_quantity)).pow(-1)
            #print(probability)
            y = pyro.sample("y", dist.Binomial(10, probability).to_event(1))
     
           
            return y
    
    return model


num_steps, start_lr, end_lr = 1000, 0.1, 0.001
ys = torch.tensor([])
ls = torch.tensor([])
eigs = torch.tensor([])
history = [(prior_mean, prior_sd)]
pyro.clear_param_store()
current_model = make_model(prior_mean, prior_sd)

for experiment in range(4):
    print("Round", experiment + 1)

    optimizer = pyro.optim.ExponentialLR({'optimizer': torch.optim.Adam,
                                          'optim_args': {'lr': start_lr},
                                          'gamma': (end_lr / start_lr) ** (1 / num_steps)})

    eig = nmc_eig(current_model, l, observation_labels=["y"], target_labels=["theta"], N=100, M=10)

    best_eig = torch.max(eig).unsqueeze(0)
    eigs = torch.cat([eigs, best_eig], dim=0)
    
    best_l_index = torch.argmax(eig)
    best_l = l[best_l_index].clone()   #shape (d,)
    best_l = best_l.unsqueeze(0)       #shape(1,d)

    y = generate_real(best_l)    #used this earlier for svi already shape should be (1,1)

    ls = torch.cat([ls, best_l], dim=0)
    ys = torch.cat([ys, y], dim=0)

    conditioned_model = pyro.condition(model, {"y": ys})

    svi = SVI(conditioned_model,
              guide,
              Adam({"lr": .005}),
              loss=Trace_ELBO(),
              num_samples=100)
    num_iters = 2000
    for i in range(num_iters):
        elbo = svi.step(ls)

    history.append((
                    pyro.param("posterior_mean").detach().clone(),
                    pyro.param("posterior_sd").detach().clone()))

    current_model = make_model(pyro.param("posterior_mean").detach().clone(),
                               pyro.param("posterior_sd").detach().clone())
    
# final posterior has (mean, sd) = (tensor([4.2266]), tensor([0.1311]))

"""ys
Out[133]: 
tensor([[3.],
        [2.],
        [8.],
        [4.]])

ls
Out[134]: 
tensor([[5.0000],
        [5.2000],
        [5.4000],
        [3.3000]])
"""
############################################################################

#Saving experiment's results as numpy arrays / pickle objects

eig_np = eigs.detach().numpy()   
ls_np = ls.detach().numpy()
ys_np = ys.detach().numpy()

path1= r'C:\Users\angus\OneDrive\Desktop\thesis\code\experiments\mice_experiment\eig.npy'
path2= r'C:\Users\angus\OneDrive\Desktop\thesis\code\experiments\mice_experiment\design.npy'
path3 = r'C:\Users\angus\OneDrive\Desktop\thesis\code\experiments\mice_experiment\y.npy'
np.save(path1, eig_np)
np.save(path2, ls_np)
np.save(path3, ys)

import pickle
with open(r'C:\true_path\thesis\code\experiments\mice_experiment\history.pickle', 'wb') as handle:
    pickle.dump(history, handle)
    
with open(r'C:\true_path\code\experiments\mice_experiment\history.pickle', 'rb') as handle:
    b = pickle.load(handle)
    
################################################################################

''' Plotting the posterior distribution of theta for the BOED choice of experiments and the 
 sub-optimal [2,3,4,5] choice'''   
    
import matplotlib
import matplotlib.pyplot as plt    
from scipy.stats import norm
import matplotlib.colors as colors
import matplotlib.cm as cmx

fig, ax1 = plt.subplots(1, 1)
x = np.arange(2.,7.,0.01)
y1 = norm.pdf(x, 3.5, 0.75)
y2 = norm.pdf(x, 4.47, 0.21)
y3 = norm.pdf(x, 4.2266, 0.1311)
ax1.axvline(x=4.27, color = 'red', linestyle='--', label = 'true $\\theta$')
ax1.plot(x, y1, label = 'prior')
ax1.plot(x, y2, label = 'sub-optimal posterior')
ax1.plot(x, y3, label = 'BOED posterior')

ax1.legend(loc="upper left", fontsize=9)
#ax1.set(xlabel="$\\theta$", ylabel='p.d.f.')
ax1.set_xlabel("$\\theta$", fontsize=9)
ax1.set_ylabel('p.d.f.', fontsize=9) 
ax1.tick_params(axis="x", labelsize=9)
ax1.tick_params(axis="y", labelsize=9)



