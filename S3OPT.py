# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 20:52:23 2025

@author: XwGu
"""

import numpy
import scipy
import math

class S3OPT:
    def __init__(self):
        self.H=0
        self.Lambda=1.1
        self.theta=math.pi/18*5
        self.layer={}
        self.layer[self.H]={}
        self.radius={}
        self.radius[self.H]=2*(1-math.cos(math.pi/3))
        
    def train(self,data,y):
        data=data/numpy.sqrt(numpy.sum(data**2,axis=1)).reshape(-1,1)
        L,W=data.shape
        self.protopical_init(data,y)
        datatemp=data[0,:].reshape(1,-1)
        labeltemp=int(y[0])
        self.layer[0]['prototype']=datatemp
        self.layer[0]['label']=[labeltemp]
        self.layer[0]['purity']=[1]
        self.layer[0]['support']=[1]
        self.layer[0]['linkage']=[0]
        self.layer[0]['NP']=1
        for ii in range(1,L):
            datatemp0=data[ii,:].reshape(1,-1)
            labeltemp0=int(y[ii])
            self.treegrowth(datatemp0,labeltemp0)
            
    def SSLtrain(self,data,y,uldata):
        data=data/numpy.sqrt(numpy.sum(data**2,axis=1)).reshape(-1,1)
        self.protopical_init(data,y)
        L,W=data.shape
        datatemp=data[0,:].reshape(1,-1)
        labeltemp=int(y[0])
        self.layer[0]['prototype']=datatemp
        self.layer[0]['label']=[labeltemp]
        self.layer[0]['purity']=[1]
        self.layer[0]['support']=[1]
        self.layer[0]['linkage']=[0]
        self.layer[0]['NP']=1
        for ii in range(1,L):
            datatemp0=data[ii,:].reshape(1,-1)
            labeltemp0=int(y[ii])
            self.treegrowth(datatemp0,labeltemp0)          
        uldata=uldata/numpy.sqrt(numpy.sum(uldata**2,axis=1)).reshape(-1,1) 
        U,W=uldata.shape
        SL=0
        while SL==0 and U!=0:
            ye,soc=self.test(uldata)
            soc0=numpy.sort(soc,axis=1)
            tempseq0=soc0[:,-1]-soc0[:,-2]*self.Lambda
            tempseq1=numpy.array([i for i in range(0,U) if tempseq0[i]>0],dtype='int32')
            tempseq2=numpy.array([i for i in range(0,U) if tempseq0[i]<=0],dtype='int32')
            if len(tempseq1)==0:
                SL=1
            else:
                pldata=uldata[tempseq1,:].copy().reshape(-1,W)
                pl=ye[tempseq1].copy()
                self.protopical_update(pldata,pl,soc0[tempseq1,-1])
                for ii in range(0,len(tempseq1)):
                    datatemp0=pldata[ii,:].reshape(1,-1)
                    labeltemp0=int(pl[ii])
                    self.treegrowth(datatemp0,labeltemp0)
            if len(tempseq2)==0:
                U=0
            else:
                uldata=uldata[tempseq2,:].copy().reshape(-1,W) 
                U,W=uldata.shape
                if U!=len(tempseq2):
                    print('Error')

    
    def treegrowth(self,datatemp,labeltemp):
        satis=0
        tempH=0
        tempL=0
        while satis==0:
            tempseq=[i for i in range(0,self.layer[tempH]['NP']) if self.layer[tempH]['linkage'][i]==tempL]
            dist0=self.cdist(self.layer[tempH]['prototype'][tempseq,:],datatemp)
            idx=numpy.argmin(dist0)
            tempdist=numpy.min(dist0)**2
            if tempdist>self.radius[tempH]:
                self.layer[tempH]['prototype']=numpy.append(self.layer[tempH]['prototype'],datatemp,axis=0)
                self.layer[tempH]['label']=numpy.append(self.layer[tempH]['label'],labeltemp)
                self.layer[tempH]['purity']=numpy.append(self.layer[tempH]['purity'],1)
                self.layer[tempH]['support']=numpy.append(self.layer[tempH]['support'],1)
                self.layer[tempH]['NP']=self.layer[tempH]['NP']+1
                self.layer[tempH]['linkage']=numpy.append(self.layer[tempH]['linkage'],tempL)
                satis=1
            else:
                tempL=tempseq[idx]
                tempprototype_orig=self.layer[tempH]['prototype'][tempL,:].copy().reshape(1,-1)
                tempsupport_orig=self.layer[tempH]['support'][tempL]
                templabel_orig=self.layer[tempH]['label'][tempL]
                temppurity_orig=self.layer[tempH]['purity'][tempL]         
                self.layer[tempH]['support'][tempL]=tempsupport_orig+1
                tempprototype_upda=(tempprototype_orig*tempsupport_orig+datatemp)/self.layer[tempH]['support'][tempL]
                tempprototype_upda=tempprototype_upda/numpy.sqrt(numpy.sum(tempprototype_upda**2,axis=1)).reshape(-1,1)
                self.layer[tempH]['prototype'][tempL,:]=tempprototype_upda
                
                tempsupport_upda=self.layer[tempH]['support'][tempL]
                if self.layer[tempH]['purity'][tempL]==1:
                    if self.layer[tempH]['label'][tempL]!=labeltemp:
                        self.layer[tempH]['label'][tempL]=-1
                        self.layer[tempH]['purity'][tempL]=0
                        templabel_upda=-1
                        temppurity_upda=0
                        growth=1
                        tempH0=tempH
                        while growth==1:
                            tempH0=tempH0+1
                            tempRadius=2*(1-math.cos(self.theta/(1.2*tempH0)))
                            if tempdist<=tempRadius:
                                growth=1
                            else:
                                growth=0
                            if tempH0>self.H:
                                self.radius[tempH0]=tempRadius
                                self.layer[tempH0]={}
                                self.H=self.H+1
                                if growth==1:
                                    self.layer[tempH0]['prototype']=tempprototype_upda.reshape(1,-1)
                                    self.layer[tempH0]['label']=[templabel_upda]
                                    self.layer[tempH0]['purity']=[temppurity_upda]
                                    self.layer[tempH0]['support']=[tempsupport_upda]
                                    self.layer[tempH0]['linkage']=[tempL]
                                    self.layer[tempH0]['NP']=1
                                    tempL=self.layer[tempH0]['NP']-1
                                else:
                                    self.layer[tempH0]['prototype']=datatemp
                                    self.layer[tempH0]['label']=[labeltemp]
                                    self.layer[tempH0]['purity']=[1]
                                    self.layer[tempH0]['support']=[1]
                                    self.layer[tempH0]['linkage']=[tempL]                
                                    self.layer[tempH0]['prototype']=numpy.append(self.layer[tempH0]['prototype'],tempprototype_orig,axis=0)
                                    self.layer[tempH0]['label']=numpy.append(self.layer[tempH0]['label'],templabel_orig)
                                    self.layer[tempH0]['purity']=numpy.append(self.layer[tempH0]['purity'],temppurity_orig)
                                    self.layer[tempH0]['support']=numpy.append(self.layer[tempH0]['support'],tempsupport_orig)
                                    self.layer[tempH0]['linkage']=numpy.append(self.layer[tempH0]['linkage'],tempL) 
                                    self.layer[tempH0]['NP']=2                            
                            else:
                                if growth==0:
                                    self.layer[tempH0]['prototype']=numpy.append(self.layer[tempH0]['prototype'],datatemp,axis=0)
                                    self.layer[tempH0]['support']=numpy.append(self.layer[tempH0]['support'],1)
                                    self.layer[tempH0]['linkage']=numpy.append(self.layer[tempH0]['linkage'],tempL)
                                    self.layer[tempH0]['label']=numpy.append(self.layer[tempH0]['label'],labeltemp)
                                    self.layer[tempH0]['purity']=numpy.append(self.layer[tempH0]['purity'],1)
                                    
                                    self.layer[tempH0]['prototype']=numpy.append(self.layer[tempH0]['prototype'],tempprototype_orig,axis=0)
                                    self.layer[tempH0]['label']=numpy.append(self.layer[tempH0]['label'],templabel_orig)
                                    self.layer[tempH0]['purity']=numpy.append(self.layer[tempH0]['purity'],temppurity_orig)
                                    self.layer[tempH0]['support']=numpy.append(self.layer[tempH0]['support'],tempsupport_orig)
                                    self.layer[tempH0]['linkage']=numpy.append(self.layer[tempH0]['linkage'],tempL)
                                    
                                    self.layer[tempH0]['NP']=self.layer[tempH0]['NP']+2
                                else:
                                    self.layer[tempH0]['prototype']=numpy.append(self.layer[tempH0]['prototype'],tempprototype_upda,axis=0)
                                    self.layer[tempH0]['support']=numpy.append(self.layer[tempH0]['support'],tempsupport_upda)
                                    self.layer[tempH0]['linkage']=numpy.append(self.layer[tempH0]['linkage'],tempL)
                                    self.layer[tempH0]['label']=numpy.append(self.layer[tempH0]['label'],templabel_upda)
                                    self.layer[tempH0]['purity']=numpy.append(self.layer[tempH0]['purity'],temppurity_upda)
                                    self.layer[tempH0]['NP']=self.layer[tempH0]['NP']+1
                                    tempL=self.layer[tempH0]['NP']-1
                        satis=1
                    else:
                        satis=1
                else:
                    satis=0
                    tempH=tempH+1
                    tempL=tempseq[idx]
                        
                
    def test(self,data):
        data=data/numpy.sqrt(numpy.sum(data**2,axis=1)).reshape(-1,1)
        L,W=data.shape
        ye=numpy.zeros((L,))
        leafprototype=numpy.empty((0,W))
        leaflabel=numpy.empty((0,))
        leafsup=numpy.empty((0,))
        for ii in range(0,self.H+1):
            tempseq=[i for i in range(0,self.layer[ii]['NP']) if self.layer[ii]['purity'][i]==1]
            if len(tempseq)>0:
                leafprototype=numpy.append(leafprototype,self.layer[ii]['prototype'][tempseq,:],axis=0)
                leaflabel=numpy.append(leaflabel,self.layer[ii]['label'][tempseq])
                leafsup=numpy.append(leaflabel,self.layer[ii]['support'][tempseq])
        seqcl=numpy.unique(leaflabel)
        cl=len(seqcl)
        soc=numpy.zeros((L,cl))
        for ii in range(0,cl):
            tempseq=[i for i in range(0,len(leaflabel)) if leaflabel[i]==seqcl[ii]]
            LPpc=leafprototype[tempseq,:].copy().reshape(-1,W)
            temp1=numpy.exp(-1*self.cdist(data,LPpc)**2)
            tempseq=numpy.argmax(temp1,axis=1)
            soctemp=numpy.max(temp1,axis=1)*leafsup[tempseq]+numpy.exp(-1*self.cdist(data,self.miu[ii,:].reshape(1,-1))**2).reshape(-1,)*self.ss[ii]
            soc[:,ii]=soctemp/(leafsup[tempseq]+self.ss[ii])
        ye=numpy.argmax(soc,axis=1)
        return ye,soc

            
    def cdist(self,data0,data1):
        temp=scipy.spatial.distance.cdist(data0,data1,'euclidean')
        return temp  
        
    def protopical_init(self,data,y):
        seqcl=numpy.unique(y)
        cl=len(seqcl)
        [L,W]=data.shape
        self.miu=numpy.empty((cl,W))
        self.ss=numpy.empty((cl,))
        for ii in range(0,cl):
            data1=data[y==ii,:].copy()
            miu1=numpy.mean(data1,axis=0).reshape(1,-1)
            miu1=miu1/numpy.sqrt(numpy.sum(miu1**2,axis=1)).reshape(-1,1)
            self.miu[ii,:]=miu1
            self.ss[ii]=sum(y==ii)
            
    # def protopical_update(self,data,y,soc):
    #     seqcl=numpy.unique(y)
    #     cl=len(seqcl)
    #     [L,W]=data.shape
    #     for ii in range(0,cl):
    #         if sum(y==ii)>=1:
    #             data1=data[y==ii,:].copy()
    #             S1=soc[y==ii].reshape(-1,1)
    #             miu1=numpy.sum(data1*S1,axis=0).reshape(-1,)
    #             self.miu[ii,:]=(self.miu[ii,:]*self.ss[ii]+miu1)/[numpy.sum(S1)+self.ss[ii]]
    #             self.miu[ii,:]=self.miu[ii,:]/numpy.sqrt(numpy.sum(self.miu[ii,:]**2))
    #             self.ss[ii]=self.ss[ii]+numpy.sum(S1)
                
    def protopical_update(self,data,y,soc):
        seqcl=numpy.unique(y)
        cl=len(seqcl)
        [L,W]=data.shape
        for ii in range(0,cl):
            if sum(y==ii)>=1:
                data1=data[y==ii,:].copy()
                miu1=numpy.sum(data1,axis=0)
                self.miu[ii,:]=(self.miu[ii,:]*self.ss[ii]+miu1)/[numpy.sum(y==ii)+self.ss[ii]]
                self.miu[ii,:]=self.miu[ii,:]/numpy.sqrt(numpy.sum(self.miu[ii,:]**2))
                self.ss[ii]=self.ss[ii]+numpy.sum(y==ii)