
import numpy as np
import torch
import numpy as np
import pyro
import pyro.distributions as dist
from torch.distributions.constraints import positive
from pyro.contrib.oed.eig import nmc_eig
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam


def create_designs(x_lower=-1, x_upper=1, x_step=1, y_lower=-1, y_upper=1, y_step=1):
    '''Creates the experimental designs to evaluate'''
    x, y = torch.meshgrid([torch.arange(x_lower, x_upper, x_step), torch.arange(y_lower, y_upper, y_step)])
    cols = torch.prod(torch.tensor(x.shape))
    x = x.reshape([cols, 1])
    
    
    cols2 = torch.prod(torch.tensor(y.shape))
    y = y.reshape([cols2,1])
    
    
    z = torch.cat([x, y], axis = 1)
    return z

l = create_designs(x_step = 0.1, y_step = 0.1)

#Specifiying the parameters for the model and the prior distributions
a_prior_mean = torch.tensor([0.68])
a_prior_sd = torch.tensor([0.008])
prior_mean = torch.tensor([0.0, 0.0])
prior_sd = torch.tensor([0.2, 0.2])

m = torch.tensor([0.5])
b = torch.tensor([0.25])
a = torch.tensor([2.0])
sd = torch.tensor([0.05])
a_fixed = torch.tensor([2.2])  

    
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
#############################################################################
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
        pyro.sample("a", dist.LogNormal(a_posterior_mean, a_posterior_sd).to_event(1))
        
def guide3(l):
    # The guide is initialised at the prior
    posterior_mean = pyro.param("posterior_mean", prior_mean.clone())
    posterior_sd = pyro.param("posterior_sd", prior_sd.clone(), constraint=positive)
    
    with pyro.plate_stack("plate", l.shape[:-1]):  
        pyro.sample("theta", dist.Normal(posterior_mean, posterior_sd).to_event(1))
##############################################################################
##############################################################################       

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

##############################################################################
''' 'make_model' are wrappers around the earlier defined models that allow
    us to easily create models with updated priors during a full run of a 
    sequential experiment'''
def make_model1(prior_mean, prior_sd):
    def model1(l):
        #Model 1 represents the fully specified model
        #Given a tensor l, it returns l.shape[:-1] independent experimental 
        #outcomes y where each y_i is associated to a choice of xi_i (in this case 
        #elements of l) and a value of theta sampled from the prior
        with pyro.plate_stack("plate", l.shape[:-1]):       
        
            theta = pyro.sample("theta", dist.Normal(prior_mean, prior_sd).to_event(1))
             
            denominator = (theta - l).pow(2).sum(axis = 1) + m 
            
            mu = a * denominator.pow(-1).unsqueeze(1) + b
            
            log_mu = torch.log(mu)
            y = pyro.sample("y", dist.Normal(log_mu, sd).to_event(1))
 
        return y
    
    return model1


def make_model2(prior_mean, prior_sd, a_prior_mean, a_prior_sd):
    def model2(l):
        #Model 2 can be used for both PMU and for maximisation of the joint EIG.
        #Given a tensor l, it returns l.shape[:-1] independent experimental 
        #outcomes y where each y_i is associated to a choice of xi_i (in this case 
        #elements of l), a value of theta and a value of alpha sampled from the
        #respective priors
        with pyro.plate_stack("plate", l.shape[:-1]):       
            a = pyro.sample("a", dist.LogNormal(a_prior_mean, a_prior_sd).to_event(1))
            
            theta = pyro.sample("theta", dist.Normal(prior_mean, prior_sd).to_event(1))
             
            denominator = (theta - l).pow(2).sum(axis = 1) + m 
            
            mu = a * denominator.pow(-1).unsqueeze(1) + b
            #mu = mu.unsqueeze(1)
            log_mu = torch.log(mu)
            y = pyro.sample("y", dist.Normal(log_mu, sd).to_event(1))
 
        return y
        
    
    return model2


def make_model3(prior_mean, prior_sd):
    def model3(l):
        #Model 3 represents the misspecified model.
        #Given a tensor l, it returns l.shape[:-1] independent experimental 
        #outcomes y where each y_i is associated to a choice of xi_i (in this case 
        #elements of l) a value of theta sampled from the prior and a misspecified
        #value of alpha
        with pyro.plate_stack("plate", l.shape[:-1]):       
        
            theta = pyro.sample("theta", dist.Normal(prior_mean, prior_sd).to_event(1))
            
            denominator = (theta - l).pow(2).sum(axis = 1) + m 
            
            mu = a_fixed * denominator.pow(-1).unsqueeze(1) + b
            #mu = mu.unsqueeze(1)
            log_mu = torch.log(mu)
            y = pyro.sample("y", dist.Normal(log_mu, sd).to_event(1))
 
        return y
    
    return model3
###############################################################################

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
############################################################################

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

##############################################################################

def create_eig_averages(x, number_experiments):
    '''Given a 'full_histories' object - which is the output of multiple_runs.py -
       it returns the average EIG for each model for each time-step'''
    #x.shape = (#runs, #models, 4)    4 represents [eigs, ls, ys, history] for each model
    #returns a tensor of shape (#models, number_experiments) with the average 
    #(over all runs) EIG for each model at each time step
    
    information_gains = torch.tensor([])
    for j in range(len(x[0])): #i.e. taking each model separately
        eig = torch.zeros(number_experiments)
        for i in range(len(x)):
            #print(x[i][j][0].shape)
            eig = eig + x[i][j][0]
        eig = eig / torch.tensor(len(x))
        eig = eig.unsqueeze(0)
        information_gains = torch.cat([information_gains, eig], axis = 0)
    return information_gains
        
 
def create_mse_averages(x, y, number_experiments):
    '''Given a 'full_histories' object and a corresponding 'full_true_values' -
       which is the output of multiple_runs.py -
       it returns the RMSE for each model for each time-step'''
    mses = torch.tensor([])
    for j in range(len(x[0])):
        mse = torch.zeros(number_experiments)
        for i in range(len(x)):
            hist = x[i][j][3]
            real_val = y[i][0]
            estimated_vals = []
            for k in range(len(hist)):
                estimated_vals.append(hist[k][0])
            estimated_vals.pop(0)
            estimated_vals = torch.cat(estimated_vals,axis = 0)
            estimated_vals = torch.reshape(estimated_vals, (number_experiments,2))
            squared_error = torch.sqrt((estimated_vals - real_val).pow(2).sum(axis = 1))
            mse = mse + squared_error
        mse = mse / torch.tensor(len(x))
        mse = mse.unsqueeze(0)
        mses = torch.cat([mses, mse], axis = 0)
    return mses

        
##############################################################################

           
full_histories, full_true_values = multiple_runs(number_runs = 10, number_experiments = 20) 
eig_averages = create_eig_averages(full_histories, number_experiments = 20)
mse_averages = create_mse_averages(full_histories, full_true_values, number_experiments = 20)


##############################################################################
#Saving experiment's results as numpy arrays / pickle objects

eig_averages_np = eig_averages.detach().numpy()   
mse_averages_np = mse_averages.detach().numpy()


path1= r'C:\Users\angus\OneDrive\Desktop\thesis\code\experiments\experiment_3\eig_averages.npy'
path2= r'C:\Users\angus\OneDrive\Desktop\thesis\code\experiments\experiment_3\mse_averages.npy'
np.save(path1, eig_averages_np)
np.save(path2, mse_averages_np)

 
with open(r'C:\Users\angus\OneDrive\Desktop\thesis\code\experiments\experiment_4\full_histories.pickle', 'rb') as handle:
    b = pickle.load(handle)
    
################################################################################

mse_averages_np = np.load( r'C:\Users\angus\OneDrive\Desktop\thesis\code\experiments\experiment_3\mse_averages.npy')


import matplotlib
import matplotlib.pyplot as plt    
from scipy.stats import norm
import matplotlib.colors as colors
import matplotlib.cm as cmx

fig, ax1 = plt.subplots(1, 1)
x = np.arange(1, 21)
y0 = mse_averages_np[0]
y1 = mse_averages_np[1]
y2 = mse_averages_np[2]
y3 = mse_averages_np[3]

ax1.plot(x, y0, label = 'fully specified model')
ax1.plot(x, y1, label = 'parametric model uncertainty')
ax1.plot(x, y3, label = 'optimising for joint EIG($\\theta$, $\\gamma$)')
ax1.plot(x, y2, label = 'misspecified value of $\\gamma$')

ax1.legend(loc="upper left", fontsize=9)
#ax1.set(xlabel="$\\theta$", ylabel='p.d.f.')
ax1.set_xlabel("experiment number", fontsize=9)
ax1.set_ylabel('RMSE', fontsize=9) 
ax1.tick_params(axis="x", labelsize=9)
ax1.tick_params(axis="y", labelsize=9)
