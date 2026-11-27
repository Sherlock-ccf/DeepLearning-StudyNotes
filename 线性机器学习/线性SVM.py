import pandas as pd
import os
import torch
import random


file_path = os.path.join(os.path.dirname(__file__), 'dataset', 'SVM.csv')
data_train=pd.read_csv(file_path,header=None)
inputs=data_train.iloc[:,0:5]
outputs=data_train.iloc[:,5]
features=torch.tensor(inputs.to_numpy(dtype="float32"))
labels=torch.tensor(outputs.to_numpy(dtype="float32"))

#batch read
def data_iter(batch_size,features,labels):
    num_examples=len(features)
    indices=list(range(num_examples))
    random.shuffle(indices)
    for i in range(0,num_examples,batch_size):
        batch_indices=torch.tensor(indices[i:min(i+batch_size,num_examples)])
        yield features[batch_indices],labels[batch_indices]

#define loss
C=2
def SVM_loss(w,X,y,b):
    return 0.5*torch.sum(w**2)/len(y)+C*torch.relu(1-y.reshape(-1,1)*(torch.matmul(X,w)+b))

#define optimization algorithm
def sgd(params,lr,batch_size):
    with torch.no_grad():
        for param in params:
            param-=lr*param.grad/batch_size
            param.grad.zero_()

#define train function
def train_liner(lr,num_epochs,batch_size):
    w=torch.normal(0,0.01,size=(5,1),requires_grad=True)
    b=torch.zeros(1,requires_grad=True)
    for epoch in range(num_epochs):
        for X,y in data_iter(batch_size,features,labels):
            l=SVM_loss(w,X,y,b)
            l.sum().backward()
            sgd([w,b],lr,batch_size)
        with torch.no_grad():
            train_l=SVM_loss(w,features,labels,b)
            print(f'epoch {epoch+1},loss_train {float(train_l.mean()):f}')
            print(w,b,sep=',')
    return w,b

w,b=train_liner(0.03,20,200)
