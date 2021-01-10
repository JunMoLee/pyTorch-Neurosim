import math
import numpy as np
import torch
import copy
from conditionalupdate import conductancebasedlearningrate

## device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

def c2cweightvariation(ratio=0.015):
    return np.random.normal(0, ratio, size=(1))[0]

def InvNonlinearWeight(conductance , A, B) :
    return -A * math.log(1 - conductance / B)


def NonlinearWeight(step, A, B): 
    return B * (1 - math.exp(-step/A)) 


def Write(conductance, A, B, numberofcells,  deltaWeightNormalized, conductancestep, learningrate):
    
    

    ## get currentposition
 
    currentposition = conductance.apply_(lambda x: InvNonlinearWeight(x, A, B) )
    
    currentstep = conductancestep * deltaWeightNormalized * numberofcells /2 * learningrate    

    ## get currentstep
    currentstep = torch.round(currentstep)
    
    '''
    idx1, idx2 = conductance.shape
    
    for i in range(idx1):
        for j in range(idx2):
            conductance[i][j] = InvNonlinearWeight(conductance[i][j], A, B)
    '''
    
    ## currentposition = apply(conductance, InvNonlinearWeight, A, B)
    
    ## print(currentposition.shape)
    newconductance = currentposition + currentstep
    newconductance = newconductance.apply_(lambda x: NonlinearWeight(x, A, B) )

    '''
    idx1, idx2 = newconductance.shape

    for i in range(idx1):
        for j in range(idx2):
            newconductance[i][j] = NonlinearWeight(newconductance[i][j], A, B)
    '''
    ## newconductance = apply(newconductance, NonlinearWeight, A, B)
    
    return newconductance

class ConWriteModule:

    def __init__ (self, Gp, Gn):


        self.poslearningrate = Gp
        self.neglearningrate = Gn
       
    def setRef ( self, rrefGp, rrefGn, idx1, idx2):
        
        refGp = rrefGp
        refGn = rrefGn
        
        pos = refGn.apply_(lambda x: conductancebasedlearningrate(x))
       
        neg = refGp.apply_(lambda x: conductancebasedlearningrate(x))
        self.poslearningrate[idx1][idx2] = pos
        self.neglearningrate[idx1][idx2] = neg
        
    def ConWrite(self, conductance, posupdate, idx1, idx2, A, B, numberofcells,  deltaWeightNormalized, conductancestep, learningrate):
        

       

        
        ## get currentposition
        currentposition = conductance.apply_(lambda x: InvNonlinearWeight(x, A, B) )
        
        
        
        currentstep = conductancestep * deltaWeightNormalized * numberofcells /2


        if posupdate :
            

            currentstep = learningrate * currentstep * self.poslearningrate[idx1][idx2]

            
        else :
            
            currentstep = learningrate * currentstep * self.neglearningrate[idx1][idx2]
            
        ## get currentstep
        currentstep = torch.round(currentstep)
 
        '''
        idx1, idx2 = conductance.shape
        
        for i in range(idx1):
            for j in range(idx2):
                conductance[i][j] = InvNonlinearWeight(conductance[i][j], A, B)
        '''
        ## currentposition = apply(conductance, InvNonlinearWeight, A, B)
        
        ## print(currentposition.shape)
        newconductance = currentposition + currentstep
        newconductance = newconductance.apply_(lambda x: NonlinearWeight(x, A, B) )

        '''
        idx1, idx2 = newconductance.shape

        for i in range(idx1):
            for j in range(idx2):
                newconductance[i][j] = NonlinearWeight(newconductance[i][j], A, B)
        '''
        ## newconductance = apply(newconductance, NonlinearWeight, A, B)
        
        return newconductance


    def apply(tensor, function, A, B):
        idx1, idx2 = tensor.shape

        for i in range(idx1):
            for j in range(idx2):
                tensor[i][j] = function(tensor[i][j], A, B)

        return tensor
        
def newwrite(conductance, refconductance, A, B, numberofcells,  deltaWeightNormalized, conductancestep, learningrate):
    

   
    # refconductance = refconductance.clone()
    
    ## get currentposition
    currentposition = conductance.apply_(lambda x: InvNonlinearWeight(x, A, B) )
    
    lr = refconductance.apply_(lambda x: conductancebasedlearningrate(x))
    
    currentstep = conductancestep * deltaWeightNormalized * numberofcells /2


    currentstep = learningrate * currentstep * lr
        
    ## get currentstep
    currentstep = torch.round(currentstep)

    '''
    idx1, idx2 = conductance.shape
    
    for i in range(idx1):
        for j in range(idx2):
            conductance[i][j] = InvNonlinearWeight(conductance[i][j], A, B)
    '''
    ## currentposition = apply(conductance, InvNonlinearWeight, A, B)
    
    ## print(currentposition.shape)
    newconductance = currentposition + currentstep
    newconductance = newconductance.apply_(lambda x: NonlinearWeight(x, A, B) )

    '''
    idx1, idx2 = newconductance.shape

    for i in range(idx1):
        for j in range(idx2):
            newconductance[i][j] = NonlinearWeight(newconductance[i][j], A, B)
    '''
    ## newconductance = apply(newconductance, NonlinearWeight, A, B)
    
    return newconductance    
