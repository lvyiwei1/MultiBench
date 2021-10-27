import sys
import os
sys.path.append(os.getcwd())
import torch
import numpy as np
from skimage.color import gray2rgb,rgb2gray
device='cuda:0'
from datasets.avmnist.get_data import get_dataloader
trains,valid,test=get_dataloader('/home/paul/yiwei/avmnist/_MFAS/avmnist',no_robust=True)
from private_test_scripts.lime_idea.general_method_2 import *

def cl_preprocess(x,l):
    return torchstackunsqueeze([torch.FloatTensor(rgb2gray(i))/255.0 for i in x],0)

def in_preprocess(x):
    return gray2rgb(x.squeeze().numpy()*255.0)

from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
segmenter=SegmentationAlgorithm('quickshift',kernel_size=4,max_dist=200,ratio=0.2)
explainer = lime_image.LimeImageExplainer(random_state=0)

pt=sampling(test,40,10,True)
sampled=sampling(test,40,10,True)
#model=torch.load('bestavmnist.pt')
#model=torch.load('avmnistrefnet.pt')
#model=torch.load('avmnistadd.pt')
#model=torch.load('avmnistlrtf.pt')
#model=torch.load('avmnistmimatrix.pt')
model=torch.load('bestmfas.pt')
#model=torch.load('avmnistcca.pt')
#print(sampletestacc(model.cuda(),pt))
runmethod2(model,explainer,1,0,pt,sampled,2000,10,[repeatingunsqueeze,cl_preprocess],segmentor=segmenter,preprocess=in_preprocess)


