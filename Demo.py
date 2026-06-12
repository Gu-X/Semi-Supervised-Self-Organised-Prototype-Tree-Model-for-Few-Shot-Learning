import scipy
import numpy
from S3OPT import S3OPT
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def training_test_split(dataname,X,Y,shot,way,qurey,state):
    numpy.random.seed(state)
    L,W=X.shape
    Y=Y.reshape(-1,)
    xtra=numpy.empty((0,W))
    xtes=numpy.empty((0,W))
    ytra=numpy.empty((0,))
    ytes=numpy.empty((0,))
    seqcl=numpy.unique(Y)   
    if dataname=='WHURS':   
        seqcl=seqcl[[3,18, 17, 13, 9]]  
    cl=len(seqcl)
    seqcl=seqcl[numpy.random.permutation(cl)]
    for i in range(0,way):
        L0=sum(Y==seqcl[i])
        if qurey=='m':
            query1=L0-shot
        else:
            query1=qurey
        seq=numpy.random.permutation(L0)
        X1=X[Y==seqcl[i],:].copy()
        xtra=numpy.append(xtra,X1[seq[range(0,shot)],:],axis=0)
        xtes=numpy.append(xtes,X1[seq[range(shot,shot+query1)],:],axis=0)
        ytra=numpy.append(ytra,numpy.ones((shot,))*i)
        ytes=numpy.append(ytes,numpy.ones((query1,))*i)
    seqtra=numpy.random.permutation(xtra.shape[0])
    seqtes=numpy.random.permutation(xtes.shape[0])
    xtra=xtra[seqtra,:].copy()
    xtes=xtes[seqtes,:].copy()
    ytra=ytra[seqtra].copy()
    ytes=ytes[seqtes].copy()
    return xtra,ytra,xtes,ytes

def matdataload(dr):
    mat_contents = scipy.io.loadmat(dr)
    L1,W1=mat_contents['data0'].shape
    y0=numpy.zeros((L1,1),dtype='int')
    data0=numpy.zeros((L1,W1))
    for i in range(0,L1):
        y0[i,:]=mat_contents['y'][i]
        data0[i,:]=mat_contents['data0'][i] 
    return data0,y0

import sklearn.metrics
def performancemeas(y1,ye0): # performance measure for multi-class classification
    acc=sklearn.metrics.accuracy_score(y1,ye0)  # classification accuracy
    bacc=sklearn.metrics.balanced_accuracy_score(y1,ye0) # balanced classification accuracy
    f1=sklearn.metrics.f1_score(y1,ye0,average='weighted') # f1 scores
    mcc=sklearn.metrics.matthews_corrcoef(y1,ye0) # matthews correlation coefficient 
    return acc,bacc,f1,mcc
dataname='WHURS'

print(dataname)

data0,y= matdataload(dataname+'_test_data.mat') # load data
NN=2000
acc1=numpy.zeros(NN,)
acc2=numpy.zeros(NN,)
for ii in range(0,NN):
    xtra,ytra,xtes,ytes=training_test_split(dataname,data0,y,1,5,15,ii) # one-shot five-way with 15 query images per class
    syst=S3OPT()  # create a new prototype tree
    syst.train(xtra,ytra) # train the prototype tree in a supervised manner
    ye,soc=syst.test(xtes) # test the trained prototype tree with query images
    acc1[ii],_,_,_=performancemeas(ytes,ye) # compute erformance measures
    
    syst=S3OPT() # create a new prototype tree
    syst.SSLtrain(xtra,ytra,xtes) # train the prototype tree in a semi-supervised manner by involving query images as unlabelled training data
    ye,soc=syst.test(xtes) # test the trained prototype tree with query images (transductive learning setting)
    acc2[ii],_,_,_=performancemeas(ytes,ye) # compute erformance measures


print([numpy.mean(acc1),numpy.std(acc1)/numpy.sqrt(NN)*1.96])
print([numpy.mean(acc2),numpy.std(acc2)/numpy.sqrt(NN)*1.96])



data0,y= matdataload(dataname+'_test_data.mat') # load data
NN=2000
acc1=numpy.zeros(NN,)
acc2=numpy.zeros(NN,)
for ii in range(0,NN):
    xtra,ytra,xtes,ytes=training_test_split(dataname,data0,y,5,5,15,ii) # five-shot five-way with 15 query images per class
    syst=S3OPT()  # create a new prototype tree
    syst.train(xtra,ytra) # train the prototype tree in a supervised manner
    ye,soc=syst.test(xtes) # test the trained prototype tree with query images
    acc1[ii],_,_,_=performancemeas(ytes,ye) # compute erformance measures
    
    syst=S3OPT() # create a new prototype tree
    syst.SSLtrain(xtra,ytra,xtes) # train the prototype tree in a semi-supervised manner by involving query images as unlabelled training data
    ye,soc=syst.test(xtes) # test the trained prototype tree with query images (transductive learning setting)
    acc2[ii],_,_,_=performancemeas(ytes,ye) # compute erformance measures


print([numpy.mean(acc1),numpy.std(acc1)/numpy.sqrt(NN)*1.96])
print([numpy.mean(acc2),numpy.std(acc2)/numpy.sqrt(NN)*1.96])
