# all optimizers definitions

import torch

class SGD_main():
    """base class for SGD from which all classes will inherit"""

    def __init__(self, model, lr:float=3e-3):
      self.model = model # used to access parameters of the model
      self.lr = lr
    
    def step(self):
        """perform gradient update"""

        with torch.no_grad():
            for p in self.model.parameters():
                if not p.grad is None:
                    # update parameter inplace
                    p.sub_(self.lr * p.grad)
    
    def zero_grad(self):
        """erase the gradients"""

        for p in self.model.parameters():
            if not p.grad is None:
                p.grad.zero_()

class SGD_weight_decay(SGD_main):
    """SGD extended with weight decay"""
  
    def __init__(self, model, lr:float=3e-3, wd:float=1e-4):
        super().__init__(model, lr)
        self.wd = wd
  
    def wd_loss(self):
        """calculates portion of loss from weight decay"""

        return self.wd * self.frobenius_norm()
  
    def frobenius_norm(self):
        """get forbenius norm summed across all parameters in model"""

        fn = 0.0
        for p in self.model.parameters():
            fn += torch.sqrt(torch.sum(p**2))
        return fn

class SGD_momentum(SGD_wd):
    """SGD with momentum -- approximates first moment of gradient for smoother updates"""
  
    def __init__(self, model, lr:float=3e-3, wd:float=1e-2, m:float=.9):
        super().__init__(model, lr, wd)
        self.m = m
        self.prev_steps = [] # store momentum terms
    
    def step(self):
        """perform update step with momentum"""
    
        with torch.no_grad():
            for i, p in enumerate(self.model.parameters()):
                if not p.grad is None:
                    if i >= len(self.prev_steps):
                        m_t = self.lr * p.grad # prev step is assumed 0 initially
                        self.prev_steps.append(m_t)
                    else:
                        # calculate momentum term for update
                        m_t = (self.m * self.prev_steps[i]) + (self.lr * p.grad)
                        self.prev_steps[i] = m_t
                    p.sub_(m_t) # update parameters inplace

class RMS_prop(SGD_wd):
    """RMSProp optimizer"""
 
    def __init__(self, model, lr:float=3e-3, wd:float=1e-4, b:float=.9, e:float=1e-8):
        super().__init__(model, lr, wd)
        self.b = b
        self.epsilon = e
        self.prev_rms = [] # tracks second moment terms
    
    def step(self):
        """scales each update inversely by the sqrt of past squared gradients -- this
        is estimated with a exponentially decaying average of the past squared gradients"""
 
        with torch.no_grad():
            for i, p in enumerate(self.model.parameters()):
                if not p.grad is None:
                    if i >= len(self.prev_rms):
                        r_dx = (1-self.b) * (p.grad**2)
                        self.prev_rms.append(r_dx)
                    else:
                        # approximates second moment of gradient (variance) to scale update inversely
                        r_dx = self.b*self.prev_rms[i] + (1-self.b) * (p.grad**2)
                        self.prev_rms[i] = r_dx
                    step = (self.lr * p.grad)/torch.sqrt(r_dx + self.epsilon)
                    p.sub_(step) # update parameters inplace

class Adam(SGD_wd):
  """adam (adaptive moment estimation) optimizer"""
  
    def __init__(self, model, lr:float=3e-3, wd:float=1e-4, m:float=.9, b:float=.99, e:float=1e-8):
        super().__init__(model, lr, wd)
        self.m = m
        self.b = b
        self.epsilon = e
        self.prev_mom = []
        self.prev_rms = []
        self.t = 1.
    
    def step(self):
        """perform optimization step with adam algorithm"""
 
        with torch.no_grad():
            for i, p in enumerate(self.model.parameters()):
                if not p.grad is None:
                    if i >= len(self.prev_mom) or i >= len(self.prev_rms):
                        # first moment of gradient
                        m_t = (1-self.m) * p.grad
                        self.prev_mom.append(m_t)
            
                        # second moment of gradient
                        v_t = (1-self.b) * (p.grad**2)
                        self.prev_rms.append(v_t)
                    else:
                        # first moment of gradient
                        m_t = self.m*self.prev_mom[i] + (1-self.m) * p.grad
                        self.prev_mom[i] = m_t
          
                        # second moment of gradient
                        v_t = self.b*self.prev_rms[i] + (1-self.b) * (p.grad**2)
                        self.prev_rms[i] = v_t
            
                    # perform update
                    """NOTE: this correction must be done because m_t and v_t are initialized
                    to zero, which causes the exp decay avg (for both moments) to be biased toward
                    zero especially at the first few updates - this update here is used to correct
                    this bias -- correction becomes lesser at later iterations"""
                    m_t_hat = m_t/(1 - self.m**self.t)
                    v_t_hat = v_t/(1 - self.b**self.t)
                    step = (self.lr * m_t_hat)/(torch.sqrt(v_t_hat) + self.epsilon)
                    p.sub_(step) # update the params inplace
          
            # update iteration for bias correction
            self.t += 1
