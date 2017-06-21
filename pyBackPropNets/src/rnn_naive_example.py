'''
Created on Jun 18, 2017

@author: Francisco Dominguez
'''
"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""
#Firstly start collecting the data
import numpy as np
import microTensorFlow as utf

# data I/O
data = open('homenajepneruda0018SinAcentos.txt', 'r').read()    # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print 'data has %d characters, %d unique.' % (data_size, vocab_size)
char_to_ix = { ch:i for i, ch in enumerate(chars) }
ix_to_char = { i:ch for i, ch in enumerate(chars) }

# hyperparameters
hidden_size = 100    # size of hidden layer of neurons
seq_length = 25    # number of steps to unroll the RNN for
learning_rate = 1e-1

D=vocab_size
H=hidden_size # hidden layer neurons

#Unrolling
X  ={}
Y  ={}
NH0={}
NH ={}
NY ={}
L  ={}
def unroll():
    NH=utf.Variable(np.zeros((H,1)))
    for t in range(seq_length):
            xs = np.zeros((vocab_size, 1))    # encode in 1-of-k representation
            ys = np.zeros((vocab_size, 1))    # encode in 1-of-k representation
            X[t]=utf.Variable(xs)
            Y[t]=utf.Variable(ys)
            NH0[t]=utf.FullyConnectedLinearRnn(X[t],NH[t-1])
            NH[t]=utf.Tanh(NH0[t])
            NY[t]=utf.FullyConnectedLinear(NH[t],vocab_size)
            L[t]=utf.SoftmaxCrossEntropyLoss(Y[t],NY[t])
def setData(inputs,targets,hprev):
    for t in range(seq_length):
            xs = np.zeros((vocab_size, 1))    # encode in 1-of-k representation
            xs[inputs[t]] = 1
            ys = np.zeros((vocab_size, 1))    # encode in 1-of-k representation
            ys[targets[t]] = 1
            X[t].value=xs
            Y[t].value=ys
def forward():
    for t in range(seq_length):
        NH0[t].forward()
        NH[t] .forward()
        NY[t] .forward()
        L[t]  .forward()    
def backward():
    for t in reversed(range(seq_length)):
        L[t]  .backward()    
        NY[t] .backward()
        NH[t] .backward()
        NH0[t].backward()
def update(alpha):
    for t in reversed(range(seq_length)):
        NY[t].w.update(alpha)
        NY[t].b.update(alpha)
        NH0[t].w.update(alpha)
        NH0[t].u.update(alpha)
        NH0[t].b.update(alpha)
    
n, p = 0, 0
unroll()
while True:
    # prepare inputs (we're sweeping from left to right in steps seq_length long)
    if p + seq_length + 1 >= len(data) or n == 0: 
        hprev = np.zeros((hidden_size, 1))    # reset RNN memory
        p = 0    # go from start of data
    inputs  = [char_to_ix[ch] for ch in data[p    :p + seq_length    ]]
    targets = [char_to_ix[ch] for ch in data[p + 1:p + seq_length + 1]]
    
    setData(inputs,targets,hprev)
    forward()
    backward()
    update(learning_rate)
    hprev=NH[-1].value

    p += seq_length    # move data pointer
    n += 1    # iteration counter 





















