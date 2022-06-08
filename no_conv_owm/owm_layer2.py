import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import os
from  numpy.random import  seed
dtype = torch.cuda.FloatTensor



# os.environ["CUDA_VISIBLE_DEVICES"] = ""  # use gpu
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""
type list [784, 800]  [800, 10]
"""
shape_list = [[784, 800], [800, 10]]
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.w1 = nn.Linear(784, 800, bias=False)
        self.w2 = nn.Linear(800, 10, bias=False)
        self.P1 = Variable(torch.eye(784, 784)).to(device)
        self.P2 = Variable(torch.eye(800, 800)).to(device)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        self.P1 = torch.sub(self.P1, compute_P(self.P1, torch.mean(x, 0, keepdim=True), 0.009))
        y1 = self.relu(self.w1(x))
        self.P2 = torch.sub(self.P2, compute_P(self.P2, torch.mean(y1, 0, keepdim=True), 0.6))
        y2 = self.relu(self.w2(y1))
        #return F.log_softmax(y2, dim=1)
        return y2



    '''
    compute_delta_P: input P in task k-1, and x_mean in taks k-1, output delta_P in task k 
    '''
def compute_P(P, x_mean, alpha_rate):
    r = torch.matmul(P, torch.transpose(x_mean, 1, 0))  #r = P.x^T (784,784) (784, 1)  ->(784, 1)
    k = torch.matmul(x_mean, P)  #k = x.P (1,784)(784,784)  -> (1,784)
    delta_P = torch.div(torch.matmul(r, k), alpha_rate + torch.matmul(x_mean, r))
    return delta_P
  