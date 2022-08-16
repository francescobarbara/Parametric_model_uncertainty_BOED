import torch
#Specifiying the parameters for the model and the prior distributions
prior_mean = torch.tensor([3.5])
prior_sd = torch.tensor([0.75])
l = torch.arange(1.,7.,0.1).unsqueeze(1)
c = torch.tensor([2.9])
a_prior_mean = torch.tensor([3.1])
a_prior_sd = torch.tensor([0.5])
a_fixed = torch.tensor([3.6])  
