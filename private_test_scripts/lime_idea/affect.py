import sys
import os
print(os.getcwd())
sys.path.insert(1,os.getcwd())
import torch
import numpy as np
from skimage.color import gray2rgb,rgb2gray
device='cuda:0'
from datasets.affect.get_data import get_simple_processed_data
torch.multiprocessing.set_sharing_strategy('file_system')
#trains,valid,test=get_dataloader('/home/paul/yiwei/avmnist/_MFAS/avmnist',no_robust=True)
traindata, validdata, test = \
    get_simple_processed_data('/home/paul/MultiBench/mosei_senti_data.pkl')

from private_test_scripts.lime_idea.general_method_2 import *

def cl_preprocess(x,l):
    return torchstack([torch.FloatTensor(i) for i in x],0)

def in_preprocess(x):
    return x.numpy()

#from lime import lime_base
#from lime.wrappers.scikit_image import SegmentationAlgorithm
#segmenter=SegmentationAlgorithm('quickshift',kernel_size=4,max_dist=200,ratio=0.2)
from private_test_scripts.lime_idea.TimeSeriesExplainer import EmbeddingTimeSeriesExplainer
explainer = EmbeddingTimeSeriesExplainer()

model=torch.load('mosei_ef_best.pt')
pt=sampling(test,40,2,True)
sampled=sampling(test,40,2,True)
runmethod2(model,explainer,2,0,pt,sampled,500,2,[repeating,repeating,cl_preprocess],segmentor=None,preprocess=in_preprocess)


