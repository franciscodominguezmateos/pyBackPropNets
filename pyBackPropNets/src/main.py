'''
Created on Nov 28, 2016

@author: Francisco Dominguez
'''
import math

class node:
    def __init__(self):
        self.input=[]
        self.partialFunctions=[]
        self.output=[]
        self.partialsLocal={}
        self.partialsGlobal={}
        self.partials={}
        self.operation=None
        self.differetial=None
        self.value=0
        self.partialGlobal=1
    def getValue(self):
        return self.value
    def forward(self):
        return self.value
    def backward(self):
        self.setPartialsGlobal()
        self.setPartialGlobal()
        self.setPartialsLocal()
        self.setPartials()
    def setPartialsGlobal(self):
        for o in self.output:
            self.partialsGlobal[o]=o.getPartial(self)
    def setPartialGlobal(self):
        if len(self.partialsGlobal)==0:
            self.partialGlobal=1
            return
        t=0
        for k in self.partialsGlobal:
            t+=self.partialsGlobal[k]
        self.partialGlobal=t
    #This method should be redefined
    def setPartialsLocal(self):
        for i,n in enumerate(self.input):
            f=self.partialFunctions[i]
            self.partialsLocal[i]=f(n.value)
    def setPartials(self):
        for n in self. input:
            self.partials[n]=self.partialsLocal[n]*self.partialGlobal
    def getPartial(self,i):
        return self.partials[i]
    
class scalar(node):
    def __init__(self,s):
        node.__init__(self)
        self.value=s
        self.input.append(self)
    def setPartialsLocal(self):
        self.partialsLocal[self]=1
    def setValue(self,v):
        self.value=v
    def update(self,alpha):
        partial=self.getPartial(self)
        self.value+=alpha*partial
class sum(node):
    def __init__(self,s0,s1):
        node.__init__(self)
        self.input.append(s0)
        self.input.append(s1)
        s0.output.append(self)
        s1.output.append(self)
    def forward(self):
        s0=self.input[0].getValue()
        s1=self.input[1].getValue()
        self.value=s0+s1
        return self.value
    def setPartialsLocal(self):
        n0=self.input[0]
        n1=self.input[1]
        self.partialsLocal[n0]=1
        self.partialsLocal[n1]=1
        
class mul(node):
    def __init__(self,s0,s1):
        node.__init__(self)
        self.input.append(s0)
        self.input.append(s1)
        s0.output.append(self)
        s1.output.append(self)
    def forward(self):
        s0=self.input[0].getValue()
        s1=self.input[1].getValue()
        self.value=s0*s1
        return self.value
    def setPartialsLocal(self):
        n0=self.input[0]
        n1=self.input[1]
        self.partialsLocal[n0]=n1.getValue()
        self.partialsLocal[n1]=n0.getValue()
        
class max(node):
    def __init__(self,s0,s1):
        node.__init__(self)
        self.input.append(s0)
        self.input.append(s1)
        s0.output.append(self)
        s1.output.append(self)
    def forward(self):
        s0=self.input[0].getValue()
        s1=self.input[1].getValue()
        if s0>s1:
            self.value=s0
        else:
            self.value=s1
        return self.value
    def setPartials(self):
        n0=self.input[0]
        n1=self.input[1]
        s0=n0.getValue()
        s1=n1.getValue()
        if s0>s1:
            self.partialsLocal[n0]=1
            self.partialsLocal[n1]=0
        else:
            self.partialsLocal[n0]=0
            self.partialsLocal[n1]=1
        
class sigmoid(node):
    def __init__(self,n0):
        node.__init__(self)
        self.input.append(n0)
        n0.output.append(self)
    def sigmoid(self,x):
        return 1/(1+math.exp(-x))
    def dsigmoid(self,x):
        s=self.sigmoid(x)
        return s*(1-s)
    def forward(self):
        n0=self.input[0]
        v0=n0.getValue()
        self.value=self.sigmoid(v0)
        return self.value
    def setPartialsLocal(self):
        n0=self.input[0]
        v0=self.getValue()
        self.partialsLocal[n0]=self.dsigmoid(v0)
if __name__ == '__main__':
    x=scalar(3)
    y=scalar(-4)
    z=scalar(2)
    w=scalar(-1)
    sy=sigmoid(y)
    print(sy.forward())
    sx=sigmoid(x)
    print(sx.forward())
    mulxy=mul(x,y)
    print(mulxy.forward())
    maxzw=max(z,w)
    print(maxzw.forward())
    summulmax=sum(mulxy,maxzw)
    print(summulmax.forward())
    # 1 Layer Neural Netork
    x0=scalar(1)
    x1=scalar(2)
    x2=scalar(3)
    w0=scalar(4)
    w1=scalar(3)
    w2=scalar(2)
    m0=mul(x0,w0)
    print(m0.forward())
    m1=mul(x1,w1)
    print(m1.forward())
    m2=mul(x2,w2)
    print(m2.forward())
    s01=sum(m0,m1)
    print(s01.forward())
    s012=sum(s01,m2)
    print(s012.forward())
    nn=sigmoid(s012)
    print(nn.forward())
    nn.backward()
    #loss=
    print(nn.dsigmoid(nn.getValue()))
    print(nn.partialGlobal)
    print(nn.partialsLocal[s012])
    print(nn.partials[s012])
    s012.backward()
    s01.backward()
    m2.backward()
    m1.backward()
    m0.backward()
    x2.backward()
    x1.backward()
    x0.backward()
    w2.backward()
    w1.backward()
    w0.backward()
    print(x0.getPartial(x0))
    print(x1.getPartial(x1))
    print(x2.getPartial(x2))
    print(w0.getPartial(w0))
    print(w1.getPartial(w1))
    print(w2.getPartial(w2))
    #mor general net
    network=[]
    weights=[]
    network.append(x0)
    network.append(x1)
    network.append(x2)
    network.append(w0)
    weights.append(w0)
    network.append(w1)
    weights.append(w1)
    network.append(w2)
    weights.append(w2)
    network.append(m0)
    network.append(m1)
    network.append(m2)
    network.append(s01)
    network.append(s012)
    network.append(nn)
    for n in network:
        print(n.forward())
    for n in reversed(network):
        n.backward()
    for n in network:
        if isinstance(n,scalar):
            print(n.getPartial(n))
    for w in weights:
        w.update(0.1)
    for n in network:
        print(n.forward())
    for n in reversed(network):
        n.backward()
    for n in network:
        if isinstance(n,scalar):
            print(n.getPartial(n))
            
    
    