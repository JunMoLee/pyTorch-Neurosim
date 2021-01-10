import torch

class MyMSE(torch.autograd.Function):
    @staticmethod
    def forward(self, y_pred, y):
        self.save_for_backward(y_pred,y)
        sumloss = torch.sum((y_pred-y).pow(2), dim=-1)
        sumloss = torch.sum(sumloss, dim=-1)/sumloss.shape[0]
        return torch.sum((y_pred-y).pow(2), dim=-1)

    @staticmethod
    def backward(self, grad_output):
        
        yy_pred, yy = self.saved_tensors

        grad_input = - 2*(yy-yy_pred)
    
        return grad_input, None

class MyMSEclass(torch.nn.Module):
    def __init__(self, y_pred, y):
        super(MyMSEclass, self).__init__()
        self.MSE = MyMSE()
        self.y_pred= y_pred
        self.y = y
    def forward(self, y_pred, y):
        return self.MSE(self.y_pred, self.y)
