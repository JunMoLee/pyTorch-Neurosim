import torch
import torch.optim as optim
from weightupdatecurve import Write

from weightupdatecurve import newwrite
from getParam import getParamA
from getParam import getParamB
import numpy as np
import copy


class GpGnSGD(optim.Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).
    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, moment um=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf
    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.
        Considering the specific case of Momentum, the update can be written as
        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v
        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.
        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form
        .. math::
             v = \rho * v + lr * g \\
             p = p - v
        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=0.1, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, batchsize= 1, refreshperiod = 100, gp_lr=0.1,gn_lr=0.1):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(GpGnSGD, self).__init__(params, defaults)

        ## define conductance step, nonlinearity, A, B
        ## self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = 'cpu'
        
        
        GpNL = 1
        GpNLneg = -9 
        GnNL = 1
        GnNLneg = -9

        ## refenceperiod for conductance based update
        
        self.referenceperiod = 2
        self.refreshperiod = refreshperiod
        ## quantize

        ADCbit = 8
        
        self.cellnumbers = 2 # determine number of cells
        
        self.Gpcs = 128 ## cs stands for conductance steps
        self.Gncs = 128 ## cs stands for conductance steps
        self.Gpcsrev = 128 ## cs stands for conductance steps
        self.Gncsrev = 128 ## cs stands for conductance steps
        
        self.GpA = getParamA(GpNL) * self.Gpcs
        self.GpB = getParamB(self.Gpcs, self.GpA)
        self.GpAneg = getParamA(GpNLneg) * self.Gpcs
        self.GpBneg = getParamB(self.Gpcs, self.GpAneg)
        
        self.GnA = getParamA(GnNL) * self.Gncs
        self.GnB = getParamB(self.Gncs, self.GnA)
        self.GnAneg = getParamA(GnNLneg) * self.Gncs
        self.GnBneg = getParamB(self.Gncs, self.GnAneg)

        
        self.Gp = []
        self.Gn = []
        self.posref = []
        self.negref = []
        self.conductancebasedupdate = 1
         
        ## set different learning rate for each layers

        self.learningrate = [0.3/ batchsize, 0.15/ batchsize] ## assume that there is no bias

        ## set different learning rate for each layers
        if self.conductancebasedupdate ==1 :
            self.learningratereverse = [1/ batchsize, 1/2/ batchsize]
        else : 
             self.learningratereverse = [1/ batchsize, 0.2/ batchsize] ## assume that there is no bias
        

        
        self.gp_lr = gp_lr
        self.gn_lr = gn_lr

        
        
        for idx1, group in enumerate(self.param_groups):
            self.Gp.append([])
            self.Gn.append([])
            self.posref.append([])
            self.negref.append([])

            
                
            
            for idx2, p in enumerate(group['params']):
                
                ## custom initializion (for neurosim implementation)
                # np.random.seed(100)
                p.data = torch.Tensor(( (np.random.randint(777783, size=p.data.cpu().numpy().shape) % 7 ) - 3 ) / 3).to(self.device)
                '''
                for idx, group in enumerate(p.data):
                    p.data[idx] = torch.ones_like(p.data[idx])*idx/400
                '''
                
                p.data.requires_grad_(True)
                pdatacopy = p.data.clone()
                Gpinit = torch.where(p.data>0,p.data,torch.zeros_like(p.data))*10
                Gninit = -torch.where(pdatacopy<0,pdatacopy,torch.zeros_like(pdatacopy))*10
                self.Gp[idx1].append(Gpinit)
                self.Gn[idx1].append(Gninit)
                self.posref[idx1].append(Gpinit)
                self.negref[idx1].append(Gninit)
        '''
        self.CWM = ConWriteModule(copy.deepcopy(self.Gp),copy.deepcopy(self.Gn))
        
        for idx1, group in enumerate(self.param_groups):

              
            for idx2, p in enumerate(group['params']):
                
                self.CWM.setRef(self.Gp[idx1][idx2].clone().cpu(),self.Gn[idx1][idx2].clone().cpu(), idx1, idx2)
       '''         
                
            
        
    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, iteration=0, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for idx1, group in enumerate(self.param_groups):
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            
            for idx2, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                d_p = p.grad.data
                
                ## print(d_p.data)
                ## print(d_p)
                
                
                d_ppos =torch.where(d_p>0,d_p,torch.zeros_like(d_p))
                d_pneg =torch.where(d_p<0,d_p,torch.zeros_like(d_p))

                # momentum (should be modified according to neuromorphic behavior - not completed)
                
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = d_p.clone()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                        
                        d_ppos =torch.where(d_p>0,d_p,torch.zeros_like(d_p))
                        d_pneg =torch.where(d_pcopy<0,d_pcopy,torch.zeros_like(d_pcopy))
                    
                    else:
                        d_p = buf
                       
                        d_ppos =torch.where(d_p>0,d_p,torch.zeros_like(d_p))
                        d_pneg =torch.where( d_p<0, d_p,torch.zeros_like( d_p))
                '''
                if 1 :
                    self.CWM.setRef(self.Gp[idx1][idx2].clone().cpu(), self.Gn[idx1][idx2].clone().cpu(), idx1, idx2)
                '''  
                d_pneg = -d_pneg # for positive update

    
                ### update weight (normal update)


                '''     
                
                self.Gp[idx1][idx2] = Write(self.Gp[idx1][idx2].cpu(), self.GpA, self.GpB, self.cellnumbers, d_pneg.cpu(), self.Gpcs, self.learningrate[idx2])

                # note that you should convert tensors to cpu for Write() variables 
                # apply_ only supports cpu
                
                ## self.Gp[idx1][idx2] = Write(self.Gp[idx1][idx2], self.GpAneg, self.GpBneg, 2, -d_ppos, self.Gpcs, group['lr'])

                ## self.Gp[idx1][idx2] -= group['lr'] * d_pneg
                self.Gp[idx1][idx2] = torch.clamp(self.Gp[idx1][idx2], max=1)
                
                
                
                
                
                ## print(self.Gp[idx1][idx2])

                self.Gn[idx1][idx2] =  Write(self.Gn[idx1][idx2].cpu(), self.GnA, self.GnB, self.cellnumbers, d_ppos.cpu(), self.Gncs, self.learningrate[idx2])

                # note that you should convert tensors to cpu for Write() variables
                # apply_ only supports cpu

                
                ## self.Gn[idx1][idx2] = Write(self.Gn[idx1][idx2], self.GnAneg, self.GnBneg, 2, -d_pneg, self.Gncs, group['lr'])
                ## self.Gn[idx1][idx2] += group['lr'] * d_ppos
                self.Gn[idx1][idx2] = torch.clamp(self.Gn[idx1][idx2], max=1)
                ## print(self.Gn[idx1][idx2])
                ## p.data.add_(-1, p.data - (self.Gp[idx1][idx2]-self.Gn[idx1][idx2]))
                '''

                ### reverse update

                reverseupdate = iteration%2
                normalupdate = 0< iteration%8000 and iteration%8000<100
                
                if iteration%2 == 10: ## or use referenceperiod 
                    self.negref[idx1][idx2] =self.Gn[idx1][idx2].clone().cpu()
                    self.posref[idx1][idx2] = self.Gp[idx1][idx2].clone().cpu()


                

                if reverseupdate == 0 or normalupdate == 1:
                    self.Gp[idx1][idx2] = Write(self.Gp[idx1][idx2].clone(), self.GpA, self.GpB, self.cellnumbers, d_pneg.clone().cpu(), self.Gpcs, self.learningrate[idx2])
                    
                    self.Gn[idx1][idx2] =  Write(self.Gn[idx1][idx2].clone(), self.GnA, self.GnB, self.cellnumbers, d_ppos.clone().cpu(), self.Gncs, self.learningrate[idx2])
                    self.Gp[idx1][idx2] = torch.clamp(self.Gp[idx1][idx2], max=10)
                    self.Gn[idx1][idx2] = torch.clamp(self.Gn[idx1][idx2], max=10)

                else :

                    # minus update
                    self.Gp[idx1][idx2] = newwrite(self.Gp[idx1][idx2].clone(),self.negref[idx1][idx2],self.GpAneg, self.GpBneg, self.cellnumbers, -d_ppos.clone().cpu(), self.Gpcsrev, self.learningratereverse[idx2])
                    # self.refGn[idx1][idx2].cpu(),self.GpAneg,
                    # plus update
                    self.Gn[idx1][idx2] =  newwrite(self.Gn[idx1][idx2].clone(),self.posref[idx1][idx2],  self.GnAneg, self.GnBneg, self.cellnumbers, -d_pneg.clone().cpu(), self.Gncsrev, self.learningratereverse[idx2])
                    self.Gp[idx1][idx2] = torch.clamp(self.Gp[idx1][idx2],  min=0)
                    self.Gn[idx1][idx2] = torch.clamp(self.Gn[idx1][idx2], min=0)
                    


                p.data = (self.Gp[idx1][idx2] - self.Gn[idx1][idx2])/10
                p.data.requires_grad_(True)

        return loss


    def refresh(self):
        """Performs refresh
        """
        for idx1, group in enumerate(self.param_groups):
            
            for idx2, p in enumerate(group['params']):
                
  
                pdatacopy = p.data.clone()
                posweight = torch.where(p.data>0,p.data,torch.zeros_like(p.data))
                negweight = -torch.where(pdatacopy<0,pdatacopy,torch.zeros_like(pdatacopy))


                ## ideal write

                self.Gp[idx1][idx2] = posweight *10
                self.Gn[idx1][idx2] = negweight *10

                ## realistic write
                '''
                self.Gp[idx1][idx2] = Write(torch.zeros_like(posweight), self.GnA, self.GnB, self.cellnumbers, posweight, self.Gpcs, 1)
                self.Gn[idx1][idx2]  = Write(torch.zeros_like(negweight), self.GnA, self.GnB, self.cellnumbers, negweight, self.Gncs, 1)
                '''

                
                p.data = ( self.Gp[idx1][idx2] - self.Gn[idx1][idx2] ) /10
                p.data.requires_grad_(True)

    def postprocess(self):
        """Performs processing of conductance/data
        """
        for idx1, group in enumerate(self.param_groups):
            for idx2, p in enumerate(group['params']):
                break
        


