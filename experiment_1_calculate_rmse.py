import numpy as np
import torch
import pyro

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
            estimated_vals = torch.tensor(estimated_vals)
            squared_error = torch.sqrt((estimated_vals - real_val).pow(2))
            mse = mse + squared_error
        mse = mse / torch.tensor(len(x))
        mse = mse.unsqueeze(0)
        mses = torch.cat([mses, mse], axis = 0)
    return mses