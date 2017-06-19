'''
Created on Jun 17, 2017

@author: Francisco Dominguez
'''
import numpy as np

class Node:
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
class Variable(Node):
    def __init__(self,s):
        Node.__init__(self)
        self.value=s
        self.input.append(self)
    def setPartialsLocal(self):
        self.partialsLocal[self]=1 #np.ones_like(self.value)
    def setValue(self,v):
        self.value=v
    def update(self,alpha):
        partial=self.getPartial(self)
        iValue=-partial #Want the opposite direction of the gradient
        self.value+=alpha*iValue
class Weights(Node):
    def __init__(self,(sr,sc)):
        Node.__init__(self)
        self.value=np.matrix(np.random.randn(sr, sc)/np.sqrt(sc))
        self.input.append(self)
    def setPartialsLocal(self):
        self.partialsLocal[self]=1 #np.ones_like(self.value)
    def setValue(self,v):
        self.value=v
    def update(self,alpha):
        partial=self.getPartial(self)
        iValue=-partial #Want the opposite direction of the gradient
        self.value+=alpha*iValue        
class Add(Node):
    def __init__(self,s0,s1):
        Node.__init__(self)
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
class Sub(Node):
    def __init__(self,s0,s1):
        Node.__init__(self)
        self.input.append(s0)
        self.input.append(s1)
        s0.output.append(self)
        s1.output.append(self)
    def forward(self):
        s0=self.input[0].getValue()
        s1=self.input[1].getValue()
        self.value=s0-s1
        return self.value
    def setPartialsLocal(self):
        n0=self.input[0]
        n1=self.input[1]
        self.partialsLocal[n0]=1
        self.partialsLocal[n1]=-1
class Mul(Node):
    def __init__(self,s0,s1):
        Node.__init__(self)
        self.input.append(s0)
        self.input.append(s1)
        s0.output.append(self)
        s1.output.append(self)
    def forward(self):
        s0=self.input[0].getValue()
        s1=self.input[1].getValue()
        self.value=s0*s1
        return self.value
    def setPartials(self):
        n0=self.input[0]
        n1=self.input[1]
        self.partials[n0]=self.partialGlobal*self.partialsLocal[n0]
        self.partials[n1]=self.partialsLocal[n1]*self.partialGlobal
    def setPartialsLocal(self):
        n0=self.input[0]
        n1=self.input[1]
        self.partialsLocal[n0]=n1.getValue().T
        self.partialsLocal[n1]=n0.getValue().T
# POINTWISE Nodes
#PointwiseMul doesn't work 
class PointwiseMul(Node):
    def __init__(self,s0,s1):
        Node.__init__(self)
        self.input.append(s0)
        self.input.append(s1)
        s0.output.append(self)
        s1.output.append(self)
    def forward(self):
        s0=self.input[0].getValue()
        s1=self.input[1].getValue()
        self.value=np.multiply(s0*s1)
        return self.value
    def setPartialsLocal(self):
        n0=self.input[0]
        n1=self.input[1]
        self.partialsLocal[n0]=n1.getValue()
        self.partialsLocal[n1]=n0.getValue()      
class Maximum(Node):
    def __init__(self,s0,s1):
        Node.__init__(self)
        self.input.append(s0)
        self.input.append(s1)
        s0.output.append(self)
        s1.output.append(self)
    def forward(self):
        s0=self.input[0].getValue()
        s1=self.input[1].getValue()
        self.value=np.max(s0,s1)
        return self.value
    def setPartials(self):
        n0=self.input[0]
        n1=self.input[1]
        s0=n0.getValue()
        s1=n1.getValue()
        bmax =    s0>s1
        bmaxn=not bmax
        self.partialsLocal[n0]=np.int(bmax)
        self.partialsLocal[n1]=np.int(bmaxn)
class ReLU(Node):   
    def __init__(self,s0):
        Node.__init__(self)
        self.input.append(s0)
        s0.output.append(self)
    def forward(self):
        s0=self.input[0].getValue()
        self.value=np.maximum(0,s0)
        return self.value
    def setPartials(self):
        n0=self.input[0]
        s0=n0.getValue()
        bmax =s0>0
        self.partialsLocal[n0]=np.int(bmax)
class Sigmoid(Node):
    def __init__(self,n0):
        Node.__init__(self)
        self.input.append(n0)
        n0.output.append(self)
    def sigmoid(self,x):
        return 1.0/(1.0+np.exp(-x))
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
        #maybe it need to be np.multiply
        self.partialsLocal[n0]=v0*(1-v0)      
class Softmax(Node):
    def __init__(self,n0):
        Node.__init__(self)
        self.input.append(n0)
        n0.output.append(self)
    def softmax(self,l):
        #trick to avoid numerical instability
        l-=np.max(l,axis=0)
        e=np.exp(l)
        s=np.sum(e,axis=0)
        p=e/s
        return p
    #this operation is too expensive that is
    #because softmax is not used before a cross entropy Node
    #It is better use the next Node to train
    #this Node is use only to predict
    def dsoftmax(self,l):
        p=self.softmax(l)#we could reuse self.getValue(). But...
        #Jacobian is a negated outer product of p except
        #diagonal. Diagonal in this case is similar to sigmoid
        p_l=-p*p.T 
        pd=np.multiply(p,(1-p))
        p_l[np.diag_indices_from(p_l)]=pd
        return p_l
    def forward(self):
        n0=self.input[0]
        v0=n0.getValue()
        self.value=self.softmax(v0)
        return self.value
    def setPartialsLocal(self):
        n0=self.input[0]
        v0=n0.getValue()
        #maybe it need to be np.multiply
        self.partialsLocal[n0]=self.dsoftmax(v0)   
class SoftmaxCrossEntropyLoss(Node):
    def __init__(self,y,l):   
        Node.__init__(self)
        self.input.append(y) # one hot encoding binary target data
        self.input.append(l) # logits  (linear ouput)
        y.output.append(self)
        l.output.append(self)
    def softmax(self,l):
        #trick to avoid numerical instability
        l-=np.max(l,axis=0)
        e=np.exp(l)
        s=np.sum(e,axis=0)
        p=e/s
        return p
    def forward(self):
        yv=self.input[0].getValue() # input value 0
        lv=self.input[1].getValue() # input value 1 (logit)
        N=yv.shape[1]    # column number
        p=self.softmax(lv) # softmax probabilities/categorical distribution
        cross= np.multiply(yv,np.log(p))
        loss =-np.sum(cross)/N
        self.value=loss
        return self.value
    def setPartialsLocal(self):
        yn=self.input[0] # input Node 0
        ln=self.input[1] # input Node 1
        yv=yn.getValue() # input value 0
        lv=ln.getValue() # input value 1
        N=yv.shape[1]     # column number
        pv=self.softmax(lv) # softmax probabilities/categorical distribution
        L_l=pv-yv       #partial with respect to the logits.
        self.partialsLocal[ln]=L_l/N # should I use divide by N?
        L_y=-np.log(pv) #partial with respect to y. Don't used
        self.partialsLocal[yn]=L_y/N 
class MeanSquareLoss(Node):
    def __init__(self,y,l):
        Node.__init__(self)
        self.input.append(y)
        self.input.append(l)
        y.output.append(self)
        l.output.append(self)
    def forward(self):
        yv=self.input[0].getValue() #input value 0
        lv=self.input[1].getValue() #input value 1 predictions
        N=yv.shape[1]    # column number = number of data
        dif=yv-lv
        loss=dif*dif.T #/N
        self.value=loss
        return self.value
    def setPartialsLocal(self):
        yn=self.input[0] # input Node 0
        ln=self.input[1] # input Node 1
        yv=yn.getValue() # input value 0
        lv=ln.getValue() # input value 1
        N=yv.shape[1]     # column number
        dif=yv-lv
        self.partialsLocal[ln]=-2*dif #/N # should I use divide by N?
        L_y=2*dif/N #partial with respect to y. Don't used
        self.partialsLocal[yn]=L_y/N 
class MeanAbsLoss(Node):
    def __init__(self,y,l):
        Node.__init__(self)
        self.input.append(y)
        self.input.append(l)
        y.output.append(self)
        l.output.append(self)
    def forward(self):
        y=self.input[0].getValue() # input value 0
        l=self.input[1].getValue() # input value 1 predictions
        N=y.shape[1]    # column number = number of data
        dif=y-l
        loss=np.sum(np.abs(dif))/N
        self.value=loss
        return self.value
    def setPartialsLocal(self):
        yn=self.input[0] # input Node 0
        ln=self.input[1] # input Node 1
        yv=yn.getValue() # input value 0
        lv=ln.getValue() # input value 1
        N=yv.shape[1]    # column number
        dif=yv-lv
        print("dif=",dif.shape)
        dLoss=np.sign(dif) # should I use divide by N?
        self.partialsLocal[ln]= -dLoss
        self.partialsLocal[yn]=  dLoss

def forward():
    print(p.forward())
    print(L.forward())
    
def backward():
    L.backward()
    p.backward()
    w.backward()
    
if __name__ == '__main__':
    alpha=0.01
    x=Variable(np.matrix([[ 1.0, 1.0, 1.0],
                          [-2.0, 4.5, 6.2]]))
    y=Variable(np.matrix( [ 7.0, 9.0,18.0]))
    w=Variable(np.matrix([5.0,1.2]))
    #w=Weights((1,2))
    p=Mul(w,x)
    #L=MeanSquareLoss(y,p)
    L=MeanAbsLoss(y,p)
    forward()
    backward()
    print(p.getPartial(w))
    w.update(alpha) 
    forward()
    backward()
    print(p.getPartial(w))
    w.update(alpha)      
    forward()
    backward()
    print(p.getPartial(w))
    w.update(alpha)      
    forward()
    backward()
    print(p.getPartial(w))
    w.update(alpha)     