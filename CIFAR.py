import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
import matplotlib as plt 
import matplotlib_inline.backend_inline

matplotlib_inline.backend_inline.set_matplotlib_formats('svg')



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
transform = T.Compose([T.ToTensor(), T.Normalize([.5,.5,.5],[.5,.5,.5])])
trainset = torchvision.datasets.CIFAR10(root = './data' , train =  True , download = True , transform = transform)
devset = torchvision.datasets.CIFAR10(root = './data' , train =  True , download = True , transform = transform)

randidx = np.random.permutation(100000)
devset = Subset(devset,randidx[:6000])
testset = Subset(devset,randidx[6000:])

batchsize = 32
train_loader = DataLoader(trainset,batch_size=batchsize , shuffle=True,drop_last=True)
dev_loader = DataLoader(devset , batch_size=batchsize)
test_loader = DataLoader(testset , batch_size=len(testset))
print(f"Number of training samples: {len(trainset)}")
print(len(trainset))
print('\nData categories:')
print(trainset.classes)

X,y = next(iter(train_loader))
print( X.data.shape)
def myNet(printtoggle = False):

    class cnn(nn.Module):
        def __init__(self,printtoggle):
            super().__init__()

            self.print = printtoggle

            self.conv1 = nn.Conv2d(3,64,3 , padding=1)
            self.bnorm1 = nn.BatchNorm2d(64)


            self.conv2 = nn.Conv2d(64,128,3)
            self.bnorm2 = nn.BatchNorm2d(128)

            self.conv3 = nn.Conv2d(128,256,3)
            self.bnorm3 = nn.BatchNorm2d(256)

            self.fc1 = nn.Linear(2*2*256,128)
            self.fc2 = nn.Linear(128,64)
            self.fc3 = nn.Linear(64,10)


        def forward(self,x):
            if self.print: print(f'Input: {list(x.shape)}')

            x = F.max_pool2d(self.conv1(x),2)
            x = F.leaky_relu(self.bnorm1(x))
            
            x = F.max_pool2d(self.conv2(x),2)
            x = F.leaky_relu(self.bnorm2(x))

            x = F.max_pool2d(self.conv3(x),2)
            x = F.leaky_relu(self.bnorm3(x))
            #x = x.view(-1, 256 * 2 * 2)


            nunits = x.shape.numel()/x.shape[0]
            x = x.view(-1,int(nunits))
            if self.print: print(f'Vectorized: {list(x.shape)}')


            x = F.leaky_relu(self.fc1(x))
            x = F.dropout(x, p=0.5 , training = self.training)
            x = F.leaky_relu(self.fc2(x))
            x = F.dropout(x,p=.5,training=self.training) # training=self.training means to turn off during eval mode
            x = self.fc3(x)
            if self.print: print(f'Final output: {list(x.shape)}')

            return x

    net = cnn(printtoggle)
    lossfun = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(),lr = 0.001, weight_decay = 1e-5)

    return net , lossfun , optimizer








net,lossfun,optimizer = myNet(True)

X,y = next(iter(train_loader))
yHat = net(X)
print('\nOutput size:')
#print(yHat.shape)
loss = lossfun(yHat,torch.squeeze(y))
print(' ')
print('Loss:')
print(loss)

def trainfunc():
    numepochs = 10

    net,lossfun,optimizer = myNet()
    net.to(device)


    trainLoss = torch.zeros(numepochs)
    devLoss = torch.zeros(numepochs)
    trainAcc = torch.zeros(numepochs)
    devAcc = torch.zeros(numepochs)


    for epochi in range(numepochs):

        net.train()
        batchLoss = []
        batchAcc = []

        for X,y in train_loader:
            X = X.to(device)
            y = y.to(device)

      # forward pass and loss
            yHat = net(X)
            loss = lossfun(yHat,y)

      # backprop
            optimizer.zero_grad()
        loss.backward()
        optimizer.step()

      # loss and accuracy from this batch
        batchLoss.append(loss.item())
        batchAcc.append( torch.mean((torch.argmax(yHat,axis=1) == y).float()).item() )
    # end of batch loop...

    # and get average losses and accuracies across the batches
    trainLoss[epochi] = np.mean(batchLoss)
    trainAcc[epochi]  = 100*np.mean(batchAcc)


    #### test performance (here done in batches!)
    net.eval() # switch to test mode
    batchAcc  = []
    batchLoss = []
    for X,y in dev_loader:

      # push data to GPU
      X = X.to(device)
      y = y.to(device)

      # forward pass and loss
      with torch.no_grad():
        yHat = net(X)
        loss = lossfun(yHat,y)
      
      # loss and accuracy from this batch
      batchLoss.append(loss.item())
      batchAcc.append( torch.mean((torch.argmax(yHat,axis=1) == y).float()).item() )
    # end of batch loop...

    # and get average losses and accuracies across the batches
    devLoss[epochi] = np.mean(batchLoss)
    devAcc[epochi]  = 100*np.mean(batchAcc)

  # end epochs

  # function output
    return trainLoss,devLoss,trainAcc,devAcc,net


trainloss , devloss ,trainAcc , devAcc,net = trainfunc()

net.eval() # switch to test mode
X,y = next(iter(test_loader))

# push data to GPU
X = X.to(device)
y = y.to(device)

# forward pass and loss
with torch.no_grad():
  yHat = net(X)
  loss = lossfun(yHat,y)

# loss and accuracy from this batch
testLoss = loss.item()
testAcc  = 100*torch.mean((torch.argmax(yHat,axis=1) == y).float()).item()



fig,ax = plt.subplots(1,2,figsize=(16,5))

ax[0].plot(trainLoss,'s-',label='Train')
ax[0].plot(devLoss,'o-',label='Dev')
ax[0].plot(len(devLoss)-1,testLoss,'r*',markersize=15,label='Test')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss (CEL)')
ax[0].set_title('Model loss')

ax[1].plot(trainAcc,'s-',label='Train')
ax[1].plot(devAcc,'o-',label='Dev')
ax[1].plot(len(devAcc)-1,testAcc,'r*',markersize=15,label='Test')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Accuracy (%)')
ax[1].set_title(f'Final model dev/test accuracy: {devAcc[-1]:.2f}/{testAcc:.2f}%')
ax[1].legend()

plt.show()