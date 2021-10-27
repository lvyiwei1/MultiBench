from lime import lime_base
import numpy as np
import copy
import random
from sklearn.utils import check_random_state
from scipy import spatial
class StaticExplainer:
    def __init__(self,kernelfn=None,feature_selection='auto',verbose=True):
        if kernelfn is None:
            def kernelfn(d):
                return np.sqrt(np.exp(-(d ** 2) / 0.25 ** 2))
        self.base=lime_base.LimeBase(kernelfn,verbose)
        self.fs = feature_selection
    def explain_instance(self,inp,classfn,correct, samples,totallabels, seed=0, fracs=2):
        randomstate=check_random_state(seed)
        masks=randomstate.randint(0,fracs+1,(samples)*len(inp)).reshape(samples,len(inp)).astype(np.float64)
        masks /= float(fracs)
        #print(samples)
        distances = np.zeros(samples)
        labels=np.zeros((samples,totallabels))
        datas = np.zeros((samples,len(inp)))
        for i in range(samples):
            if i==0 or (np.sum(masks[i])==0.0):
                datas[i]=inp
                distances[i]=0.0
                masks[i]=np.ones(len(inp))
            else:   
                datas[i]=inp*masks[i]
                distances[i]=spatial.distance.cosine(masks[0],masks[i])
            labels[i]=classfn(datas[i])
        return self.base.explain_instance_with_data(masks,labels,distances,correct,len(inp),feature_selection=self.fs)
            

