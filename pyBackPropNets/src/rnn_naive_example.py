'''
Created on Jun 18, 2017

@author: Francisco Dominguez
'''
"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""
#Firstly start collecting the data
import pickle, gzip
import numpy as np
import matplotlib.pyplot as plt
import mainMatrix as utf

# data I/O
data = open('homenajepneruda0018SinAcentos.txt', 'r').read()    # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print 'data has %d characters, %d unique.' % (data_size, vocab_size)
char_to_ix = { ch:i for i, ch in enumerate(chars) }
ix_to_char = { i:ch for i, ch in enumerate(chars) }

NV=valid_set[0].shape[0]
N=train_set[0].shape[0]
#N=500
D=train_set[0].shape[1]
H=100 # hidden layer neurons
train_set_x=addOnesFirstRow(np.matrix(train_set[0]).T)[:,:N]
valid_set_x=addOnesFirstRow(np.matrix(valid_set[0]).T)
#one hot encoding
train_set_y=np.matrix(np.zeros((10,N)))
train_set_y[train_set[1][:N],np.arange(N)]=1
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
h=utf.Variable(np.matrix(np.zeros((H,N))))
#hidden layer (rnn naive node)
Wxh=utf.Weights((H,D))
Whh=utf.Weights((H,H))
bh=utf.Variable((np.matrix(np.zeros((H,1)))))
linXH=utf.Mul(Wxh,x)
linHH=utf.Mul(Whh,h)
linH0=utf.Add(linXH,linHH)
linH =utf.Add(linH0,bh)
h=utf.Tanh(linH)
#output layer
Why=utf.Weights((D,H))
by=utf.Variable((np.matrix(np.zeros((D,1)))))
linHY=utf.Mul(Why,h)
linY=utf.Add(linHY,by)
L=utf.SoftmaxCrossEntropyLoss(y,linY)

def forward():
    lin.forward()
    h1.forward()
    lin2.forward()
    L.forward()

def backward():
    L.backward()
    lin2.backward()
    h1.backward()
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
a=utf.Softmax(lin2)
p=a.forward()
q=np.argmax(p,axis=0)
i=(train_set[1]==q)*1
# print("q=",np.array(q[:5]))
# print("y=",train_set[1][:5])
# print("i=",np.array(i[:5]))
print("training   accuracy=",np.float(np.sum(i))/N)
#Validation data evaluation
x.value=valid_set_x
lin.forward()
h1.forward()
lin2.forward()
p=a.forward()
q=np.argmax(p,axis=0)
i=(valid_set[1]==q)*1
# print("q=",np.array(q[:5]))
# print("y=",valid_set[1][:5])
# print("i=",np.array(i[:5]))
print("validation accuracy=",np.float(np.sum(i))/NV)
w=w.value
#print(w.shape)
for i in range(10):
    plt.subplot(2, 5, i+1)
    imgw=w[i,1:]
    plt.imshow(imgw.reshape(28,28), cmap="gray_r")
plt.show()




















