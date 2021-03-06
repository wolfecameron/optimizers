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

class SGD_momentum(SGD_main):
    """SGD with momentum -- approximates first moment of gradient for smoother updates"""
  
    def __init__(self, model, lr:float=3e-3, m:float=.9):
        super().__init__(model, lr)
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

class SGD_demon(SGD_main):
    """SGD momentum implemented to match https://arxiv.org/pdf/1910.04952.pdf"""

    def __init__(self, model, lr:float=3e-3, m:float=0.9):
        super().__init__(model, lr)
        self.m = m
        self.prev_steps = []

    def step(self):
        """implements a different recursion for momentum than the one
        given above"""

        with torch.no_grad():
            for i, p in enumerate(self.model.parameters()):
                if not p.grad is None:
                    if i >= len(self.prev_steps):
                        p.sub_(self.lr*p.grad)
                        v_t = -self.lr*p.grad
                        self.prev_steps.append(v_t)
                    else:
                        update = self.lr*p.grad - self.m*self.prev_steps[i]
                        p.sub_(update)
                        v_t = self.m*self.prev_steps[i] - self.lr*p.grad
                        self.prev_steps[i] = v_t

class RMS_prop(SGD_main):
    """RMSProp optimizer"""
 
    def __init__(self, model, lr:float=3e-3, b:float=.9, e:float=1e-8):
        super().__init__(model, lr)
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

class Adagrad(SGD_main):
    """adaptive gradient optimizer"""

    def __init__(self, model, lr:float=3e-3, e:float=1e-8):
        super().__init__(model, lr)
        self.epsilon = e
        self.sum_sq_grad = []

    def step(self):
        """perform update using adaptive gradient update step"""

        with torch.no_grad():
            for i, p in enumerate(self.model.parameters()):
                if not p.grad is None:
                    if i >= len(self.sum_sq_grad):
                        sq_grad = (p.grad**2)
                        self.sum_sq_grad.append(sq_grad)
                    else:
                        self.sum_sq_grad[i] += (p.grad**2)
                        sq_grad = self.sum_sq_grad[i]
                    # algorithm performs much worse without sqrt because denominator term
                    # that scales the update grows much too quickly
                    step = (self.lr / torch.sqrt(sq_grad + self.epsilon)) * p.grad
                    p.sub_(step) # update the parameters inplace
    
class Adadelta(SGD_main):
    """adadelta gradient optimizer -- adadelta does not require a learning rate because it
    simply uses the unit correction term in the numerator instead of learning rate"""

    def __init__(self, model, m:float=0.9, e:float=1e-8):
        super().__init__(model, 0)
        self.m = m
        self.epsilon = e
        self.unit_correction = []
        self.prev_rms = []

    def step(self):
        with torch.no_grad():
            for i, p in enumerate(self.model.parameters()):
                if not p.grad is None:
                    if i >= len(self.prev_rms) or i >= len(self.unit_correction):
                        v_t = (1 - self.m) * (p.grad**2)
                        self.prev_rms.append(v_t)
                        step = (torch.sqrt(torch.tensor(self.epsilon))/torch.sqrt(v_t + self.epsilon))*p.grad
                        correction = (1 - self.m) * (step**2)
                        self.unit_correction.append(correction)
                    else:
                        v_t = self.m*self.prev_rms[i] + (1 - self.m) * (p.grad**2)
                        self.prev_rms[i] = v_t
                        step = (torch.sqrt(self.unit_correction[i] + self.epsilon)/torch.sqrt(v_t + self.epsilon))*p.grad
                        correction = (self.m)*self.unit_correction[i] + (1 - self.m)*(step**2)
                        self.unit_correction[i] = correction
                    p.sub_(step) # update parameters inplace
                         

class Adam(SGD_main):
    """adam (adaptive moment estimation) optimizer"""
  
    def __init__(self, model, lr:float=3e-3, m:float=.9, b:float=.99, e:float=1e-8):
        super().__init__(model, lr)
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

class Adam_demon(SGD_main):
    """implements adam to be used with demon https://arxiv.org/pdf/1910.04952.pdf"""
  
    def __init__(self, model, lr:float=3e-3, m:float=.9, b2:float=.99, e:float=1e-8):
        super().__init__(model, lr)
        self.m = m
        self.b2 = b2
        self.epsilon = e
        self.prev_m = []
        self.prev_sm = []
        self.t = 1.
    
    def step(self):
        """perform optimization step with adam algorithm"""

        with torch.no_grad():
            for i, p in enumerate(self.model.parameters()):
                if not p.grad is None:
                    if i >= len(self.prev_m) or i >= len(self.prev_sm):
                        # first moment
                        m_t = p.grad
                        self.prev_m.append(m_t)

                        # second moment
                        sm_t = (1. - self.b2)*(p.grad**2)
                        self.prev_sm.append(sm_t)
                    else:
                        # first moment
                        m_t = p.grad + self.m*self.prev_m[i]
                        self.prev_m[i] = m_t

                        # second moment
                        sm_t = self.b2*self.prev_sm[i] + (1. - self.b2)*(p.grad**2)
                        self.prev_sm[i] = sm_t

                    # perform the update using first and second moment
                    m_t_hat = m_t/(1 - self.m**self.t)
                    sm_t_hat = sm_t/(1 - self.b2**self.t)
                    step = (self.lr * m_t_hat)/(torch.sqrt(sm_t_hat + self.epsilon))
                    p.sub_(step) # update the params inplace

        # update iteration for bias correction 
        self.t += 1


class AdamW(SGD_main):
    """adam (adaptive moment estimation) optimizer"""
  
    def __init__(
            self, model, lr:float=3e-3, m:float=.9, b:float=.99,
            wd:float=.01, e:float=1e-8):
        super().__init__(model, lr)
        self.m = m
        self.b = b
        self.wd = wd
        self.epsilon = e
        self.prev_mom = []
        self.prev_rms = []
        self.t = 1.
    
    def step(self):
        """performs both weight decay and optimization step"""

        with torch.no_grad():
            # weight decay
            for i, p in enumerate(self.model.parameters()):
                if not p.grad is None:
                    p.sub_(self.lr*self.wd*p)

            # normal gradient update
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


class SGDW(SGD_main):
    """SGD with decoupled weight decay"""

    def __init__(self, model, lr:float=3e-3, m:float=.9, wd:float=0.01):
        super().__init__(model, lr)
        self.m = m
        self.wd = wd
        self.prev_steps = [] # store momentum terms
    
    def step(self):
        """perform update step with momentum"""
    
        with torch.no_grad():
            # decoupled weight decay term
            for i, p in enumerate(self.model.parameters()):
                if not p.grad is None:
                    p.sub_(self.lr*self.wd*p)

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
