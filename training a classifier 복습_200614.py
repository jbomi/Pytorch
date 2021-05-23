# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import torch
import torchvision
import torchvision.transforms as transforms

# +
transform=transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
#transforms.Compose 데이터 전처리 패키지
#transforms.ToTensor 데이터 타입을 tensor형태로 변경
#transfroms.Normalize() 이미지의 경우 픽셀 값 하나는 0~255값을 가짐
# 하지만 ToTensor()로 타입 변경시 0~1사이 값으로 바뀜
#transforms.Normalize((),())를 이용하여 -1~1사이 값으로 normalized 시킴
"""Normalize 두가지 연산"""
"""scaling: 데이터의 scale을 줄여줌"""
"""centering: 데이터의 중심을 원점으로 맞춰줌"""
"""ToTensor()를 해주면 scaling을 해주고"""
"""Normalize를 해주면 centering+rescaling를 해준것"""
trainset=torchvision.datasets.CIFAR10(root='./data',train=True,
                                                 download=True,transform=transform)
#root:경로지정, train: train or test 데이터를 받아옴
#transform: 사전에 설정해놓은 데이터 처리 형태
#download 데이터 셋이 없을때
trainloader=torch.utils.data.DataLoader(trainset,batch_size=4,
                                       shuffle=True,num_workers=2)
#dataset: 불러올 데이터 셋
#batch_size=batch단위 만큼 데이터를 뽑아옴
#shuffle: 데이터 shuffle 할것인가?
"""?: num_workers의 역할은?"""
testset=torchvision.datasets.CIFAR10(root='./data',train=False,
                                   download=True,transform=transform)
testloader=torch.utils.data.DataLoader(testset,batch_size=4,
                                      shuffle=False,num_workers=2)

classes=('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')
# -

# ---------------------------------------------2020_06_15

# +
import matplotlib.pyplot as plt
import numpy as np

#functions to show an image

def imshow(img):
    img=img/2+0.5 #unnormalize
    npimg=img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))#np.transpose(해당배열, 배열 크기)
    #np.transpose: 전치행렬, 행렬의 행과 열을 바꾸기
    #a.T
    #np.transpose(a)
    #np.swapaxes(a,axis1,axis2)
    plt.show()
    
#get some random training images
dataiter=iter(trainloader)
images,labels=dataiter.next()

#show images
imshow(torchvision.utils.make_grid(images))
"""?: make_grid 양옆 눈금그리는 함수인가"""
"""그림 4개만 나오게 하는게 어디서 정의되는걸까==> batch_size"""
"""grid 눈금 사이 값 어디서 정의하는걸까"""
#print labels(정답)
print(''.join('%12s' %classes[labels[j]] for j in range(4)))
"""%5s는 문자열 형태에 왜 5의 역할==>폭 지정, 5만큼의 폭을 지정하고 오른쪽 배열 """
"""그렇다면 아래 그림과 맞는 폭을 찾아보자=>>%12s"""
# -

# ----------------------------------------------2020.06.16

# # Convolutional Neural Network 정의하기

# +
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(3,6,5)
        #3 입력 이미지 채널, 6 출력 이미지 채널, 5*5 컨벌루젼 kernel
        self.pool=nn.MaxPool2d(2,2)
        # maxpool window (2,2)
        self.conv2=nn.Conv2d(6,16,5)
        #y=W(eight)x+b(ias)
        self.fc1=nn.Linear(16*5*5,120) #linear(weight,bias), 이미지 차원 5*5
        #in_features, out_features
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)
        
    def forward(self,x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        x=x.view(-1,16*5*5)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x
    
net=Net()
print(net)
# -

# # loss function과 optimizer정의하기

# +
import torch.optim as optim

criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(net.parameters(),lr=0.01,momentum=0.9)
# -

# ------------------------------------------------------------20.06.27

# # Train the network

# +
for epoch in range(2): #loop over the dataset multiple times
    running_loss=0.0
    
    for i, data in enumerate(trainloader,0):
        #get the inputs; data is a list of [inputs, labels]
        inputs, labels= data
        
        #zero the parameter gradients
        optimizer.zero_grad()
        
        #forward + backward+optimize
        outputs=net(inputs)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        
        #print statistics
        running_loss +=loss.item()
        if i %2000==1999: #print every 2000 mini=batchs
            print('[%d,%5d] loss: %.3f'%(epoch+1,i+1,running_loss/2000))
            running_loss=0.0
            
print('Finished Training')
# -

# _________________________________________ 20.06.28

PATH='./cifar_net.pth'
torch.save(net.state_dict(),PATH)

# # test the network on the test data

# +
dataiter=iter(testloader)
images,labels=dataiter.next()

#print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth : ',' '.join('%5s' %classes[labels[j]] for j in range(4)))
# -

net=Net()
net.load_state_dict(torch.load(PATH))

outputs=net(images)
