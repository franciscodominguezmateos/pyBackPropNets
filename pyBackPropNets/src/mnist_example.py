'''
Created on Jun 18, 2017

@author: Francisco Dominguez
'''
#Firstly start collecting the data
import pickle, gzip
import numpy as np
import matplotlib.pyplot as plt
import mainMatrix as utf

#this function add a row of ones to the first row of a matrix
#in order to have a bias term
def addOnesFirstRow(m):
    ones=np.ones(m.shape[1])
    rm=np.vstack([ones,m])
    return rm

# Load the dataset
# There is a train set, a validation seet and a test set.
# -Train set is to train the models
# -Validation set is to validate models
# -Test set is to test the final model
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = pickle.load(f)
f.close()

NV=valid_set[0].shape[0]
N=train_set[0].shape[0]
D=train_set[0].shape[1]
train_set_x=addOnesFirstRow(np.matrix(train_set[0]).T)
valid_set_x=addOnesFirstRow(np.matrix(valid_set[0]).T)
#one hot encoding
train_set_y=np.matrix(np.zeros((10,N)))
train_set_y[train_set[1],np.arange(N)]=1
valid_set_y=np.matrix(np.zeros((10,NV)))
valid_set_y[valid_set[1],np.arange(NV)]=1
print(train_set_y.shape)
print(train_set[1].shape)
print(train_set_x.shape)
print(train_set_x[:5,:5])
print(train_set[1][:10])
print(train_set_y[:,:10])
x=utf.Variable(train_set_x)
y=utf.Variable(train_set_y)
#plt.imshow(train_set_x[1:,7].reshape(28,28),cmap="gray_r")
#plt.show()
w=utf.Weights((10,D+1))
#Init the bias to 0 not random
w.value[:,0]=0
lin=utf.Mul(w,x)
#a=utf.Softmax(lin)
L=utf.SoftmaxCrossEntropyLoss(y,lin)

def forward():
    lin.forward()
    #a.forward()
    L.forward()

def backward():
    L.backward()
    #a.backward()
    lin.backward()
    w.backward()
alpha=0.1
i=0
while i<100:
    forward()
    backward()
    w.update(alpha)
    if i%50==0:
        iw=lin.getPartial(w)
        mi=np.min(iw)
        ma=np.max(iw)
        print("L=",L.value,"mi=",mi,"ma=",ma)   
    i+=1
print("L=",L.value,"mi=",mi,"ma=",ma)   





















