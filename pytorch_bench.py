# check https://github.com/kuangliu/pytorch-cifar
# TODO --resume
from __future__ import print_function
import argparse
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import uuid
from datetime import datetime

import numpy as np
import os
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import time
from sys import platform
import matplotlib.pyplot as plt
import json
import ssl
from functools import partial # To invoke Kernel objects with input parameters when creating KernelConv2d object (e.g. partial(GaussianKernel, 0.05) for Gaussian OR partial(PolynomialKernel,2,3) for Polynomial)
from layer import KernelConv2d, GaussianKernel, PolynomialKernel

ssl._create_default_https_context = ssl._create_unverified_context
def machine():
  return dict(linux="glx64",darwin="maci64",win32="win32").get(platform,"other")

class ConfutionMatrix:
    def __init__(self,cm):
        self.matrix = cm
    def getAccuracy(self):
        #(TP+TN)/(P+N)
        matrix = self.matrix
        sumd = np.sum(np.diagonal(matrix))
        sumall = np.sum(matrix)
        sumall = np.add(sumall,0.00000001)
        return sumd/sumall
    def getErrorRate(self):
        #(FP+FN)/(P+N)
        matrix = self.matrix
        sumd = np.sum(np.diagonal(matrix))
        sumall = np.sum(matrix)
        sumall = np.add(sumall,0.00000001)
        return (sumall-sumd)/sumall
    def getPrecision(self):
        #TP/(TP+FP)
        matrix = self.matrix
        sumrow = np.sum(matrix,axis=1)
        sumrow = np.add(sumrow,0.00000001)
        precision = np.divide(np.diagonal(matrix),sumrow)
        return np.sum(precision)/precision.shape[0]
    def getSensitivity(self): #aka recall
        #TP/P
        matrix = self.matrix
        sumcol = np.sum(matrix,axis=0)
        sumcol = np.add(sumcol,0.00000001)
        recall = np.divide(np.diagonal(matrix),sumcol)
        return np.sum(recall)/recall.shape[0]
    def getSpecificity(self):
        #TN/N
        matrix = self.matrix
        sumrow = np.sum(matrix,axis=1)
        sumrow = np.add(sumrow,0.00000001)
        spec = np.divide(np.diagonal(matrix),sumrow)
        return np.sum(spec)/spec.shape[0]
    def get2f(self):
        #2*precision*recall/(precision+recall)
        precision = self.getPrecision()
        recall = self.getSensitivity()
        return (2*precision*recall)/(precision+recall)

class NetCNN(nn.Module):
    def __init__(self,insize,inchannels,kern1=3,kern2=3,fea1=32,fea2=64,dropout1=0.25,dropout2=0.50,linear1=128):
        super(NetCNN, self).__init__()
        self.conv1 = nn.Conv2d(inchannels, fea1, kern1, 1)
        self.conv2 = nn.Conv2d(fea1, fea2, kern2, 1)
        #self.dropout1 = nn.Dropout(dropout1)
        #self.dropout2 = nn.Dropout(dropout2)
        self.pdropout1= dropout1
        self.pdropout2 =dropout2
        self.numclasses = 10
        self.linear1 = linear1

        if self.linear1 == 0:
            self.fc1 = None
            self.fc2 = nn.LazyLinear(self.numclasses)
        else:
            self.fc1 = nn.LazyLinear(self.linear1)
            self.fc2 = nn.Linear(self.linear1, self.numclasses)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),2)
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = F.dropout(x,self.pdropout1,training=self.training)
        x = torch.flatten(x, 1)
        if self.fc1 != None:
            x = self.fc1(x)
            x = F.relu(x)
            x = F.dropout(x,self.pdropout2,training=self.training)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

class NetSMAX(nn.Module):
    def __init__(self,insize,inchannels):
        super(NetSMAX, self).__init__()
        self.fc = nn.Linear(insize*inchannels, 10)

    def forward(self, x):
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output

class NetKNN(nn.Module):
    def __init__(self,insize,inchannels,fea1=10,fea2=20,kern1=5,kern2=5,linear1=50):
        super(NetKNN,self).__init__()
        self.numclasses = 10
        self.conv1=KernelConv2d(inchannels,fea1,kern1,partial(GaussianKernel, 0.05)) # self.conv1=KernelConv2d(1,10,5) for default/Ploynomial kernel with default parameters
        self.bn1=nn.BatchNorm2d(fea1)
        self.conv2=KernelConv2d(fea1,fea2,kern2)
        self.bn2=nn.BatchNorm2d(fea2)
        self.conv2_drop=nn.Dropout2d()
        self.fc1=nn.LazyLinear(linear1)
        self.fc2=nn.Linear(linear1,self.numclasses)
    def forward(self,x):
        x=F.relu(F.max_pool2d(self.conv1(x),2))
        x=self.bn1(x)
        x=F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)),2))
        x=self.bn2(x)
        x = torch.flatten(x, 1)
        #x=x.view(-1,320)
        x=F.relu(self.fc1(x))
        x=F.dropout(x,training=self.training)
        x=F.relu(self.fc2(x))
        return F.log_softmax(x,dim=1)

def train(args, model, criterion, device, train_loader, optimizer, epoch):
    model.train()
    with tqdm(train_loader, unit="batch") as tepoch:
        #for batch_idx, (data, target) in enumerate(train_loader):
        tepoch.set_description(f"Epoch {epoch}")
        for data, target in tepoch:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            tepoch.set_postfix(loss=loss.item())#, accuracy=100. * accuracy)
            if False:
                if batch_idx % args.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, criterion, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    y_pred = []
    y_true = []

    with torch.no_grad():
        for data, otarget in test_loader:
            data, target = data.to(device), otarget.to(device)
            output = model(data)
            #test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            y_pred.extend(pred.cpu().numpy()) # Save Prediction
            correct += pred.eq(target.view_as(pred)).sum().item()
            y_true.extend(otarget.cpu().numpy()) # Save Truth
    
    cf_matrix = confusion_matrix(y_true, y_pred)

    #test_loss /= len(test_loader.dataset)
    accuracyvalue = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(
        correct, len(test_loader.dataset),accuracyvalue))
    return accuracyvalue,cf_matrix

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp
def main():
    t00 = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example modified')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-gpu', action='store_true', default=False,
                        help='disables gpu training')
    parser.add_argument('--singlecore',action="store_true")
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--dataset', choices=["cifar10","mnist"], default="mnist")
    parser.add_argument('--model', choices=["cnn","softmax","knn"], default="cnn")
    parser.add_argument('--optimizer', choices=["adadelta","adam","gdescent","sgd"], default="adam")
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_gpu and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    if args.singlecore:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    else:
        #torch.set_num_threads(1)
        #torch.set_num_interop_threads(1)
        pass
    print("using intra-cor:%d inter-core:%d" % (torch.get_num_threads(),torch.get_num_interop_threads()))
    train_kwargs = {'batch_size': args.batch_size,"shuffle":True}
    test_kwargs = {'batch_size': args.test_batch_size,"shuffle":False}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)


    root = './data'
    if not os.path.exists(root):
        os.mkdir(root)

    if args.dataset == "mnist":
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (1.0,)) # mean 0.5 std 1.0
            ])
        dataset1 = datasets.MNIST(root, train=True, download=True,
                           transform=transform)
        dataset2 = datasets.MNIST(root, train=False,
                           transform=transform)
        insize = 28*28
        inchannels = 1
        classes = [str(i) for i in range(10)]
    elif args.dataset == "cifar10":
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) # mean 0.5 std 1.0
            ])
        dataset1 = datasets.CIFAR10('../data', train=True, download=True,
                           transform=transform)
        dataset2 = datasets.CIFAR10('../data', train=False,
                           transform=transform)
        insize = 32*32
        inchannels = 3
        classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    else:
        raise Exception("unknown dataset "+args.dataset)


    if args.model == "cnn":
        model = NetCNN(insize,inchannels).to(device)
    elif args.model == "softmax":
        model = NetSMAX(insize,inchannels).to(device)
    elif args.model == "knn":
        model = NetKNN(insize,inchannels).to(device)
    else:
        raise Exeption("unknown model " + args.model)

    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "gdescent":
        #raise Exception("not implemented simple gradient descent in pytorch")
        optimizer = optim.Optimizer(model.parameters(), lr=args.lr)
    elif args.optimizer == "adadelta":
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr,momentum=0.9)
    else:
        raise Exeption("unknown optimizer " + args.optimizer)

    if args.dry_run:
        dataset1 = torch.utils.data.Subset(dataset1, torch.arange(1000))
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    #torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    criterion = nn.CrossEntropyLoss()
    accuracyvalue = 0
    t = time.time()
    for epoch in range(1, args.epochs + 1):
        train(args, model, criterion, device, train_loader, optimizer, epoch)
        tt = time.time()
        accuracyvalue,cf_matrix = test(model,criterion, device, test_loader)
        test_time = time.time()-tt
        scheduler.step()
    training_time = time.time()-t

    cm = ConfutionMatrix(cf_matrix)
    go = "run_%s_%s" %(t00,uuid.uuid1())
    cm_accuracy = cm.getAccuracy()
    cm_sensitivity = cm.getSensitivity()
    cm_specificity = cm.getSpecificity()
    cm_Fscore = cm.get2f()
    total_parameters = get_n_params(model)
    iterations = len(train_loader)//args.batch_size*args.epochs
    out = dict(accuracy=float(accuracyvalue),machine=machine(),training_time=training_time,implementation="torch",single_core=1 if args.singlecore else 0,type='single',test=args.model,gpu=0 if args.no_gpu else 1,epochs=args.epochs,batch_size=args.batch_size,now_unix=time.time(),cm_accuracy=float(cm_accuracy),cm_Fscore=float(cm_Fscore),iterations=iterations,testing_time=test_time,total_params=total_parameters,cm_specificity=float(cm_specificity),cm_sensitivity=float(cm_sensitivity),args=vars(args))
    open(go+".json","w").write(json.dumps(out,indent=4))
    #np.savetxt(go+".loss.txt", losses)
    torch.save(model.state_dict(), go + ".pt")

    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = classes,
                     columns = classes)
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig(go+'.cf.png')





if __name__ == '__main__':
    main()