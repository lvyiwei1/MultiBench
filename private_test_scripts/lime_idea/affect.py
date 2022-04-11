import sys
import os
print(os.getcwd())
sys.path.insert(1,os.getcwd())
import torch
import numpy as np
from skimage.color import gray2rgb,rgb2gray
device='cuda:1'
from datasets.affect.get_data import get_simple_processed_data
torch.multiprocessing.set_sharing_strategy('file_system')
#trains,valid,test=get_dataloader('/home/paul/yiwei/avmnist/_MFAS/avmnist',no_robust=True)
traindata, validdata, test = \
    get_simple_processed_data('/home/paul/MultiBench/mosei_senti_data.pkl')

from private_test_scripts.lime_idea.general_method_3 import *

def cl_preprocess(x,l):
    return torchstack([torch.FloatTensor(i) for i in x],0)

def in_preprocess(x):
    return x.numpy()

#from lime import lime_base
#from lime.wrappers.scikit_image import SegmentationAlgorithm
#segmenter=SegmentationAlgorithm('quickshift',kernel_size=4,max_dist=200,ratio=0.2)
from private_test_scripts.lime_idea.TimeSeriesExplainer import EmbeddingTimeSeriesExplainer
explainer = EmbeddingTimeSeriesExplainer()

model=torch.load('mosei_lf_best.pt')
pt,orders=sampling(test,40,2,True)
sampled,_=sampling(test,40,2,True)
record01=runmethod2(model,explainer,0,1,pt,sampled,500,2,[cl_preprocess,repeating,repeating],segmentor=None,preprocess=in_preprocess)
record02=runmethod2(model,explainer,0,2,pt,sampled,500,2,[cl_preprocess,repeating,repeating],segmentor=None,preprocess=in_preprocess)
record10=runmethod2(model,explainer,1,0,pt,sampled,500,2,[repeating,cl_preprocess,repeating],segmentor=None,preprocess=in_preprocess)
record12=runmethod2(model,explainer,1,2,pt,sampled,500,2,[repeating,cl_preprocess,repeating],segmentor=None,preprocess=in_preprocess)
record20=runmethod2(model,explainer,2,0,pt,sampled,500,2,[repeating,repeating,cl_preprocess],segmentor=None,preprocess=in_preprocess)
record21=runmethod2(model,explainer,2,1,pt,sampled,500,2,[repeating,repeating,cl_preprocess],segmentor=None,preprocess=in_preprocess)

def average(nums):
    total=0.0
    for i in nums:
        total += i
    return total / float(len(nums))

lines=[]
for j in range(len(pt)):
    line=[]
    for i in [record01,record02,record10,record12,record20,record21]:
        line.append(i[j])
    lines.append((orders[j],line))
    print(orders[j])
    print(line)

import pickle
f=open('affect40ptdrive.pk','wb+')
pickle.dump(lines,f)


    
