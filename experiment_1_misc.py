

def generate_real(l):
    
    '''This function will be used in our main function. There we have sampled 
        a shared true value of theta, 'theta_real', and whenever a BOED approach
        picks a specific design l, we output the corresponding experimental 
        outcome using this function''' 
    
    theta_prior_mean, theta_prior_sd = theta_real, torch.tensor([0.00001])
    
          
    with pyro.plate_stack("plate", l.shape[:-1]): 
        theta = pyro.sample("theta", dist.Normal(theta_prior_mean, theta_prior_sd).to_event(1))
        distance = l - theta
        inner_quantity = a - c* (distance.pow(2))
        probability = (torch.tensor(1.) + torch.exp(inner_quantity)).pow(-1)
        y = pyro.sample("y", dist.Binomial(10, probability).to_event(1)) 
        
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
        
            theta = pyro.sample("theta",  dist.Normal(prior_mean, prior_sd).to_event(1))
            distance = l - theta
            inner_quantity = a - c* (distance.pow(2))
            probability = (torch.tensor(1.) + torch.exp(inner_quantity)).pow(-1)
            y = pyro.sample("y", dist.Binomial(10, probability).to_event(1))
               
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
        
            theta = pyro.sample("theta",  dist.Normal(prior_mean, prior_sd).to_event(1))
            a = pyro.sample("a",  dist.Normal(a_prior_mean, a_prior_sd).to_event(1))
            distance = l - theta
            inner_quantity = a - c* (distance.pow(2))
            probability = (torch.tensor(1.) + torch.exp(inner_quantity)).pow(-1)
            y = pyro.sample("y", dist.Binomial(10, probability).to_event(1))
               
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
        
            theta = pyro.sample("theta",  dist.Normal(prior_mean, prior_sd).to_event(1))
            distance = l - theta
            inner_quantity = a_fixed - c* (distance.pow(2))
            probability = (torch.tensor(1.) + torch.exp(inner_quantity)).pow(-1)
            y = pyro.sample("y", dist.Binomial(10, probability).to_event(1))
     
            return y
    
    return model3