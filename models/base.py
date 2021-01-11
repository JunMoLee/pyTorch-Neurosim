import torch
import torch.nn as nn
import torch.nn.functional as F
from quantize import quantize


## device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

inputlayer = 400
outputlayer = 10
hiddenlayer = 100
ADCbit = 8

class linearfunction(torch.autograd.Function):
    
    @staticmethod
    def forward(self, input, weight):
        self.save_for_backward(input, weight)
        v= torch.ones_like(input)
        input = torch.round(v+input) - v ## caution : 0.5 rounds to 0 so modified accordingly
        output =  input.mm(weight.t())
        

        return output
    
    @staticmethod
    def backward(self, grad_output):
        input, weight = self.saved_tensors
        grad_input = grad_output.mm(weight)
        grad_weight = grad_output.t().mm(input)
        
   
        
     
   
        return grad_input, grad_weight


class ModifiedLinear(nn.Module):
    def __init__(self, input_features, output_features):
        super(ModifiedLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        
        return linearfunction.apply(input, self.weight)



'''
class quantizelayerIH(torch.autograd.Function):
    
        
    @staticmethod
    def forward(self, input):
        
        input = quantize(input, ADCbit, inputlayer)
        ## sigmoid + quantize
        
        convertedinput = nn.Sigmoid()(input)
        self.save_for_backward(convertedinput)

        
        return input
    
    @staticmethod
    def backward(self, grad_output):

        ## sigmoid + quantize 

        
        convertedinput = self.saved_tensors[0]
        convertedgrad_output = grad_output * convertedinput
        convertedgrad_output = convertedgrad_output *(torch.ones_like(convertedinput) - convertedinput)
        
   
        return grad_output
'''
class quantizelayerIH(torch.autograd.Function):
 
    ADCbit = 8
    neurons = 400

    @staticmethod
    def forward(self, input):
        input = quantize(input, quantizelayerIH.ADCbit, quantizelayerIH.neurons)
        
        ## sigmoid + quantize
        
        
        
        convertedinput = nn.Sigmoid()( input)

        
        
        self.save_for_backward(convertedinput)

        
        return convertedinput
    
    @staticmethod
    def backward(self, grad_output):
        ## sigmoid + quantize 

        
        convertedinput = self.saved_tensors[0]
        convertedgrad_output = grad_output * convertedinput*(torch.ones_like(convertedinput) - convertedinput)
        
     
   
        return convertedgrad_output

class quantizelayerHO(torch.autograd.Function):
 
    ADCbit = 8
    neurons = 100

    @staticmethod
    def forward(self, input):
        input = quantize(input, quantizelayerHO.ADCbit, quantizelayerHO.neurons)

     
        ## sigmoid + quantize

        
        convertedinput = nn.Sigmoid()(input)

       
        self.save_for_backward(convertedinput)

        
        return convertedinput
    
    @staticmethod
    def backward(self, grad_output):
        ## sigmoid + quantize 

        
        convertedinput = self.saved_tensors[0]
        convertedgrad_output = grad_output * convertedinput*(torch.ones_like(convertedinput) - convertedinput)

     
   
        return convertedgrad_output
'''    
class quantizeIH(nn.Module):
    """
    Input: A Function
    Returns : A Module that can be used
        inside nn.Sequential
    """
    def __init__(self):
        super(quantizeIH, self).__init__()
        self.qIH = quantizelayerIH()
       
        

    def forward(self, input):
        return self.qIH.apply(input)
'''
class quantizecustomIH(nn.Module):
    """
    Input: A Function
    Returns : A Module that can be used
        inside nn.Sequential
    """
    def __init__(self):
        super(quantizecustomIH, self).__init__()
        
        self.q = quantizelayerIH()

        

    def forward(self, input):
        return self.q.apply(input)

class quantizecustomHO(nn.Module):
    """
    Input: A Function
    Returns : A Module that can be used
        inside nn.Sequential
    """
    def __init__(self):
        super(quantizecustomHO, self).__init__()
        
        self.q = quantizelayerHO()

        

    def forward(self, input):
        return self.q.apply(input)
  
class Base(nn.Module):
    

    def __init__(self, num_classes=outputlayer):
        super(Base, self).__init__()
        self.num_classes = num_classes
        self.hiddenlayer = hiddenlayer
        
        
        self.IH = ModifiedLinear(400,self.hiddenlayer)

        self.quantizeIH = quantizecustomIH()
            

        self.HO =  ModifiedLinear(self.hiddenlayer, self.num_classes)
        
        self.quantizeHO = quantizecustomHO()
        
           
        self.layerlist = [

            self.IH,
            self.quantizeIH,
            self.HO,
            self.quantizeHO,
            
            
        ]

        self.layer = torch.nn.ModuleList(self.layerlist)

    def forward(self, x):
        for unitlayer in self.layer:
            x=unitlayer(x)

        return x

