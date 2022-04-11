from lime import lime_base
import numpy as np
import copy
import torch
import random
from sklearn.utils import check_random_state
from scipy import spatial
class TextGroupExplainer:
    def __init__(self,kernelfn=None,feature_selection='auto',verbose=True):
        if kernelfn is None:
            def kernelfn(d):
                return np.sqrt(np.exp(-(d ** 2) / 0.25 ** 2))
        self.base=lime_base.LimeBase(kernelfn,verbose)
        self.fs = feature_selection
    def explain_instance(self,inp,classfn,samples,jj):
        randomstate=check_random_state(0)
        length = len(inp['text'])
        masks = np.ones((samples,length))
        sizes = randomstate.randint(1,length-2, samples-1)
        #print(samples)
        distances = np.zeros(samples)
        labels=np.zeros((samples,2))
        for i in range(samples):
            if i==0:
                data=inp
                distances[i]=0.0
                masks[i]=np.ones(length)
            else:
                inac = randomstate.choice(range(1,length-1),sizes[i-1],replace=False)
                masks[i,inac]=0
                data=copy.deepcopy(inp)
                newtext=[]
                newids=[]
                for j in range(length):
                    if masks[i][j] > 0:
                        newtext.append(data['text'][j])
                        newids.append(data['input_ids'][j])
                for j in range(len(data['input_ids'])-len(newids)):
                        newids.append(0)
                newlen = len(newtext)
                newmask = torch.zeros(len(data['input_mask']))
                newmask[0:newlen] = 1
                data['text']=newtext
                data['tokens']=newtext
                data['input_ids']=torch.LongTensor(newids)
                data['input_mask']=newmask
                labels[i]=classfn(data,jj)

                distances[i]=spatial.distance.cosine(masks[0],masks[i])
            labels[i]=classfn(data,jj)
        return self.base.explain_instance_with_data(masks,labels,distances,0,len(inp),feature_selection=self.fs),self.base.explain_instance_with_data(masks,labels,distances,1,len(inp),feature_selection=self.fs)
            

