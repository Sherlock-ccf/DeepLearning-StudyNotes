import pandas as pd
import os
import torch
import random


file_path = os.path.join(os.path.dirname(__file__), 'dataset', 'digit_recongnition.csv')
data_train=pd.read_csv(file_path)
inputs=data_train.iloc[:,1:785]
outputs=data_train.iloc[:,0]
features=torch.tensor(inputs.to_numpy(dtype="float32"))/255.0
labels=torch.tensor(outputs.to_numpy(dtype="int64"))

#batch read
def data_iter(batch_size,features,labels):
    num_examples=len(features)
    indices=list(range(num_examples))
    random.shuffle(indices)
    for i in range(0,num_examples,batch_size):
        batch_indices=torch.tensor(indices[i:min(i+batch_size,num_examples)])
        yield features[batch_indices],labels[batch_indices]

def softmax(X):
    X_exp=torch.exp(X)
    sum=X_exp.sum(1,keepdims=True)
    return X_exp/sum
#define model
def linreg(X,w,b):
    return torch.matmul(X,w)+b
def model(X,w,b):
    return softmax(linreg(X,w,b))

#define loss
def cross_entropy(y_hat,y):
    return -torch.log(y_hat[range(len(y_hat)),y])

#define optimization algorithm
def sgd(params,lr,batch_size):
    with torch.no_grad():
        for param in params:
            param-=lr*param.grad/batch_size
            param.grad.zero_()

#define train function
def train_liner(lr,num_epochs,net,loss,batch_size):
    w=torch.normal(0,0.01,size=(784,10),requires_grad=True)
    b=torch.zeros(10,requires_grad=True)
    for epoch in range(num_epochs):
        for X,y in data_iter(batch_size,features,labels):
            l=loss(net(X,w,b),y)
            l.sum().backward()
            sgd([w,b],lr,batch_size)
        with torch.no_grad():
            train_l=loss(net(features,w,b),labels)
            print(f'epoch {epoch+1},loss_train {float(train_l.mean()):f}')
    return w,b

w,b=train_liner(0.03,20,model,cross_entropy,2000)
