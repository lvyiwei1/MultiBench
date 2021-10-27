from lime import lime_base
import numpy as np
import copy
import random
from sklearn.utils import check_random_state
from scipy import spatial
class CategoricalTimeSeriesExplainer:
    def __init__(self,kernelfn=None,feature_selection='auto',verbose=False):
        if kernelfn is None:
            def kernelfn(d):
                return np.sqrt(np.exp(-(d ** 2) / 0.25 ** 2))
        self.base=lime_base.LimeBase(kernelfn,verbose)
        self.fs = feature_selection
    def explain_instance(self,inp,classfn,correct, samples,totallabels, seed=0, fracs=1):
        randomstate=check_random_state(seed)
        masks=randomstate.randint(0,fracs+1,(samples)*len(inp[0])).reshape(samples,len(inp[0])).astype(np.float64)
        masks /= float(fracs)
        #print(samples)
        distances = np.zeros(samples)
        labels=np.zeros((samples,totallabels))
        datas = np.zeros((samples,len(inp),len(inp[0])))
        for i in range(samples):
            if i==0 or (np.sum(masks[i])==0.0):
                datas[i]=inp
                distances[i]=0.0
                masks[i]=np.ones(len(inp[0]))
            else:   
                datas[i]=np.einsum("ij,j->ij",inp,masks[i])
                distances[i]=spatial.distance.cosine(masks[0],masks[i])
            labels[i]=classfn(datas[i])
        return self.base.explain_instance_with_data(masks,labels,distances,correct,len(inp[0]),feature_selection=self.fs)
            

class EmbeddingTimeSeriesExplainer:
    def __init__(self,kernelfn=None,feature_selection='auto',verbose=False):
        if kernelfn is None:
            def kernelfn(d):
                return np.sqrt(np.exp(-(d ** 2) / 0.25 ** 2))
        self.base=lime_base.LimeBase(kernelfn,verbose)
        self.fs = feature_selection
    def explain_instance(self,inp,classfn,correct, samples,totallabels, seed=0, fracs=1,framelength=5):
        #print("Explaining ")
        randomstate=check_random_state(seed)
        segments=(len(inp))//framelength
        masks=randomstate.randint(0,fracs+1,(samples)*segments).reshape(samples,segments).astype(np.float64)
        masks /= float(fracs)
        #print(samples)
        distances = np.zeros(samples)
        labels=np.zeros((samples,totallabels))
        datas = np.zeros((samples,len(inp),len(inp[0])))
        for i in range(samples):
            if i==0 or (np.sum(masks[i])==0.0):
                datas[i]=inp
                distances[i]=0.0
                masks[i]=np.ones(segments)
            else:
                #print(masks[i])
                #print(inp.shape)
                datas[i]=np.einsum("ijk,i->ijk",inp.reshape(segments,framelength,len(inp[0])),masks[i]).reshape(len(inp),len(inp[0]))
                distances[i]=spatial.distance.cosine(masks[0],masks[i])
            labels[i]=classfn(datas[i:i+1])
        #print(datas)
        #print(labels)
        return self.base.explain_instance_with_data(masks,labels,distances,correct,len(inp[0]),feature_selection=self.fs)
