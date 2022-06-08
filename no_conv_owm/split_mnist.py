from torchvision import datasets, transforms
import torch
import PIL.Image as Image
import random
import torchvision
torch.manual_seed(0)

'''
create split mnist dataset 
'''
batch_size = 40
num_task = 5

class SplitMNIST(datasets.MNIST):
    tasks = {
        0: [0, 1],
        1: [2, 3],
        2: [4, 5],
        3: [6, 7],
        4: [8, 9],
    }

    def __init__(self, root="../data", train=True, task=0, cum=True, binary=False):
        super().__init__(root, train, download=True)
        if not train and cum:
            classes = [i for t in range(task + 1) for i in SplitMNIST.tasks[t]]
        else:
            classes = [i for i in SplitMNIST.tasks[task]]
        self.idx = [i for i in range(len(self.targets)) if self.targets[i] in classes]
        self.transform = transforms.ToTensor()
        self.task = task
        self.cum = cum
        self.binary = binary
        self.train = train
        # print(classes)
        # print(self.targets)
        # print(self.targets[0] in classes)
        # print(self.idx)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, index):
        img, target = self.data[self.idx[index]], self.targets[self.idx[index]]
        img = Image.fromarray(img.numpy(), mode='L')
        img = self.transform(img)

        if (self.binary and not self.cum):
            target = target - self.task * 2
        # print(target)
        return img, target



'''
cum means cumulative, eg. task2 will contain images of 0,1,2,3
binary means labels with 0 and 1 
'''

def get_mnist():
    train_loader = {}
    train_loader_binary = {}
    test_loader = {}
    test_loader_no_cum = {}
    test_loader_binary = {}

    for i in range(num_task):
        train_loader[i] = torch.utils.data.DataLoader(SplitMNIST(train=True, task=i), 
                                                      batch_size=batch_size, drop_last=True,
                                                      num_workers=4)
        train_loader_binary[i] = torch.utils.data.DataLoader(SplitMNIST(train=True, cum=False, binary=True, task=i),
                                                      batch_size=batch_size, drop_last=True,
                                                      num_workers=4)
        test_loader[i] = torch.utils.data.DataLoader(SplitMNIST(train=False, task=i), drop_last=True,
                                                     batch_size=batch_size)
        test_loader_no_cum[i] = torch.utils.data.DataLoader(SplitMNIST(train=False, task=i, cum=False), drop_last=True,
                                                     batch_size=batch_size)
        test_loader_binary[i] = torch.utils.data.DataLoader(SplitMNIST(train=False, task=i, cum=False, binary=True),  drop_last=True,
                                                     batch_size=batch_size)
    return train_loader,train_loader_binary, test_loader, test_loader_no_cum,test_loader_binary


'''
for b, (x, y) in enumerate(torch.utils.data.DataLoader(SplitMNIST(train=True, binary=True, cum=False, task=1),
                                                       batch_size=batch_size,
                                                       num_workers=4)):
    print(y)
    print(x.size())
    print(x.dtype)
    break

train_loader,train_loader_binary, test_loader, test_loader_no_cum,test_loader_binary = get_mnist()
for i in range(5):
    print("train dataset:{}".format(i), len(train_loader[i].dataset))
    print("test dataset:{}".format(i), len(test_loader_no_cum[i].dataset))

'''