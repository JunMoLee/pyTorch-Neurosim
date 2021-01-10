import torch
import torch.nn as nn


## device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'


def quantize(tobequantized, quantizebits, numberofneurons):

    
    quantizelevel = pow(2,8) - 1
    
    maxsumvalue = numberofneurons*5  ## max value of z (f(z) = a) for a given neuron
    minsumvalue = -numberofneurons*5  ## min value of z (f(z) = a) for a given neuron
    tobequantized = tobequantized*5 ## convert to current scale
    maxvalue = maxsumvalue - minsumvalue
   
    tobequantized = torch.trunc(tobequantized / (maxvalue / quantizelevel)).type(torch.IntTensor)
   
    tobequantized =2 * tobequantized
    
   
    
   
    ## print('int')
    ## print(tobequantized)
   
    ## tobequantized = tobequantized.type(torch.FloatTensor)
    tobequantized = maxsumvalue/5  * ( tobequantized / quantizelevel )


    quantized = tobequantized

    return quantized



    
class quantizefunction(torch.autograd.Function):

    ADCbit = 8
    neurons = 100
        
    @staticmethod
    def forward(self, input):
        
        input = quantize(input, quantizefunction.ADCbit, quantizefunction.neurons)
        ## sigmoid + quantize
        '''
        convertedinput = nn.Sigmoid()(input)
        self.save_for_backward(convertedinput)

        '''
        return input.to(device)
    
    @staticmethod
    def backward(self, grad_output):

        ## sigmoid + quantize 

        '''
        convertedinput = self.saved_tensors[0]
        convertedgrad_output = grad_output * convertedinput
        convertedgrad_output = convertedgrad_output *(torch.ones_like(convertedinput) - convertedinput)
        '''
   
        return grad_output.to(device)

class quantizefunction(torch.autograd.Function):

    ADCbit = 8
    neurons = 100
        
    @staticmethod
    def forward(self, input):
        
        input = quantize(input, quantizefunction.ADCbit, quantizefunction.neurons)
        ## sigmoid + quantize
        '''
        convertedinput = nn.Sigmoid()(input)
        self.save_for_backward(convertedinput)

        '''
        return input.to(device)
    
    @staticmethod
    def backward(self, grad_output):

        ## sigmoid + quantize 

        '''
        convertedinput = self.saved_tensors[0]
        convertedgrad_output = grad_output * convertedinput
        convertedgrad_output = convertedgrad_output *(torch.ones_like(convertedinput) - convertedinput)
        '''
   
        return grad_output.to(device)

class quantizeactivation(torch.autograd.Function):


        
    @staticmethod
    def forward(self, input):
        
        input = torch.round(input)
     
        '''
        convertedinput = nn.Sigmoid()(input)
        self.save_for_backward(convertedinput)

        '''
        return input.to(device)
    
    @staticmethod
    def backward(self, grad_output):

        

        '''
        convertedinput = self.saved_tensors[0]
        convertedgrad_output = grad_output * convertedinput
        convertedgrad_output = convertedgrad_output *(torch.ones_like(convertedinput) - convertedinput)
        '''
   
        return grad_output.to(device)







