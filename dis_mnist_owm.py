import torch
import torchvision
from torch.utils.data import DataLoader
import os
#import matplotlib.pyplot as plt
from owm_layers import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ignore warning
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use gpu
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # allow 2 dll file


'''
download mnist and divide to 0-4 and 5-9
60000 for train, 10000 for test
'''
num_class = 10
epoch = 5
batch_size = 40
learning_rate = 0.01
momentum = 0.5
log_interval = 10
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./data/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size, shuffle=True)
# shuffle: random
# 0.1307 and 0.3081 are the mean and standard error of tha mnist dataset

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./data/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size, shuffle=True)


def check_figure():
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    print(example_targets)
    print(example_data.shape)

'''
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("figure {},  target {}".format(i, example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    #plt.show()

'''
check_figure()

network = Net()
network = network.to(device)
train_loader = train_loader
test_loader = test_loader

optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

train_losses = []
train_counter = []
test_losses = []


def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append((batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
            torch.save(network.state_dict(), './model.pth')
            torch.save(optimizer.state_dict(), './optimizer.pth')

def test():
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
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


for epoch in range(1, epoch + 1):
    train(epoch)
    test()

