import torch
import torchvision
from torch.utils.data import DataLoader
import os
from split_mnist import *
from owm_layer2 import *
from tensorboardX import SummaryWriter
dtype = torch.cuda.FloatTensor
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ignore warning
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # allow 2 dll file



'''
the upper bound of continual learnig, multi 
download mnist and divide to sub task
{
    1:[0,1]
    2:[2,3]
    3:[4,,5]
    4:[6,7]
    5:[8,9]

}
60000 for train, 10000 for test
'''
RUN_Task = 5
num_class = 10
epochs = 5
batch_size = 40
learning_rate = 0.01
momentum = 0.5
log_interval = 10
shape_list = [[784, 800], [800, 10]]
param_shape_list = [[800, 784], [800], [10, 800], [10]]
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

writer = SummaryWriter("../runs", comment="cum_task")


torch.cuda.empty_cache()  #清除显存

network = Net(shape_list)
network = network.to(device)




# writer.add_graph(network, torch.rand(40, 784),True)

optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
#optimizer = optim.Adam(network.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-5, weight_decay=0)







def train(epoch, train_loader):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(batch_size, 784)
        data = data.to(device)
        target = target.to(device)
        output = network(data)
        optimizer.zero_grad()
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append((batch_idx * batch_size) + ((epoch - 1) * len(train_loader.dataset)))
            torch.save(network.state_dict(), './model.pth')
            torch.save(optimizer.state_dict(), './optimizer.pth')
            if epoch == 1:
                writer.add_scalar("loss/train:", loss.item(), batch_idx)
            

def test(epoch, test_loader):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(batch_size, 784)
            data = data.to(device)
            target = target.to(device)
            output = network(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
     
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))



train_loader, train_loader_binary, test_loader, test_loader_no_cum, test_loader_binary = get_mnist()




train_losses = []
train_counter = []
test_losses = []

for task_num in range(RUN_Task):
    for epoch in range(1, epochs + 1):
        print('Train on Task:', task_num)
        train(epoch, train_loader_binary[task_num])
    print("test on all task", task_num)
    test(epoch, test_loader_binary[task_num])
    


