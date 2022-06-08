import torch
import os
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from split_mnist import *
from owm_layer2 import *
from optim_owm import SGD_OWM
from tensorboardX import SummaryWriter
dtype = torch.cuda.FloatTensor
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ignore warning
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # allow 2 dll file



'''
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
epochs = 5    # pevious epoch=5 maybe not enougth for OWM method
batch_size = 40
learning_rate = 0.01
momentum = 0 #最原始梯度下降
log_interval = 10
shape_list = [[784, 800], [800, 10]]
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




writer = SummaryWriter(comment="OWM")


torch.cuda.empty_cache()  #清除显存

network = Net()
network = network.to(device)
# writer.add_graph(network, torch.rand(40, 784),True)

#optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
#optimizer = optim.Adam(network.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-5, weight_decay=0)
optimizer = SGD_OWM(network.parameters(), lr=learning_rate, momentum=momentum)


result_log = open("results.txt", mode = 'a',encoding='utf-8')




def train(epoch, train_loader, task_num, P_list):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(batch_size, 784)
        data = data.to(device)
        target = target.to(device)
        output = network(data)
        loss = F.cross_entropy(output, target, reduction='sum')
        optimizer.zero_grad()
        loss.backward()


        optimizer.step(P_list)
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append((batch_idx * batch_size) + ((epoch - 1) * len(train_loader.dataset)))
            torch.save(network.state_dict(), './model.pth')
            torch.save(optimizer.state_dict(), './optimizer.pth')
    i = batch_idx + (epoch-1) * len(train_loader) + task_num * epochs * len(train_loader)
    # writer.add_scalar("loss/owm_train:", float(loss.item()), i)
           
              

def test(epoch, test_loader, task_num):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(batch_size, 784)
            data = data.to(device)
            target = target.to(device)
            output = network(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)), file=result_log)
        # writer.add_scalars("loss/owm_test:", test_loss)
        # writer.add_scalars("owm_accuracy:", correct/len(test_loader.dataset))


train_loader, train_loader_binary, test_loader, test_loader_no_cum, test_loader_binary = get_mnist()



#P_list =  [P1, P2] # used for updating parameters after each task finished

P_list=[Variable(torch.eye(784, 784)).to(device), Variable(torch.eye(800, 800)).to(device)]

for task_num in range(RUN_Task):
    train_losses = []
    train_counter = []
    test_losses = []
    for epoch in range(1, epochs + 1):
        print('Test on Task:', task_num, file=result_log)
        print('Train on Task:', task_num)
        train(epoch, train_loader_binary[task_num], task_num, P_list)
        print('Test on Task:', task_num)
        test(epoch, test_loader_binary[task_num], task_num)


    print("test on previous task")
    for pre_task in range(task_num + 1):
        print('trained on task:', task_num, ' test on task:', pre_task)
        print('trained on task:', task_num, ' test on task:', pre_task, file=result_log)
        test(epoch, test_loader_binary[pre_task], pre_task)

    P_list[0] = network.P1
    P_list[1] = network.P2


result_log.close()