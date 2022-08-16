import torch

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
