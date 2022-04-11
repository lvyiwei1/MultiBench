import sys
import os
import torch
import numpy as np
#from skimage.color import gray2rgb,rgb2gray
device='cuda:0'
from scipy import spatial
dataloader=torch.load("/home/paul/yiwei/mmf/hatefulmemedataloader.pt")

from mmf.common.registry import registry
from mmf.common.sample import to_device

#model_cls = registry.get_model_class("late_fusion")
#model = model_cls.from_pretrained("late_fusion.hateful_memes").to(device)

model_cls = registry.get_model_class("mmbt")
model = model_cls.from_pretrained("mmbt.hateful_memes.images").to(device)
datas=dataloader.dataset

samplefrom = []

for i in datas:
    samplefrom.append((i,i['targets'].item()))


from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
segmenter=SegmentationAlgorithm('quickshift',kernel_size=4,max_dist=200,ratio=0.2)
explainer = lime_image.LimeImageExplainer(random_state=0)

sys.path.insert(1,os.getcwd())
from private_test_scripts.lime_idea.general_method_2 import *
pt,_=sampling(samplefrom,40,2,True,totorch=False)
sampled,_=sampling(samplefrom,40,2,True,totorch=False)

#print(sampled[0][0]['image'].size())

#runmethod2(model,explainer,1,0,pt,sampled,1000,10,[repeatingunsqueeze,cl_preprocess],segmentor=segmenter,preprocess=in_preprocess)
#runmethod2(model,explainer,0,1,pt,sampled,1000,10,[cl_preprocess,repeatingunsqueeze],segmentor=segmenter,preprocess=in_preprocess)

def val2rgb(x):
    return (x+3.8272839)/0.03291828679103477

def rgb2val(x):
    return x*0.03291828679103477-3.8272839

import copy
from mmf.common.sample import convert_batch_to_sample_list
import torch.nn.functional as F
def makeclassify(base):
    def classify(inp):
        slist=[]
        for i in inp:
            q=torch.FloatTensor(rgb2val(i)).transpose(0,2)
            b=copy.deepcopy(base)
            b['image']=q
            slist.append(b)
        samplelist=convert_batch_to_sample_list(slist)
        with torch.no_grad():
            out=model(to_device(samplelist,device))['scores']
        return F.softmax(out,dim=1).detach().cpu().numpy()
    return classify

countpt=0
records=[]
for (sample,correct) in pt:
    countpt += 1
    countsampled = 0
    classify=makeclassify(sample)
    inp=val2rgb(sample['image'].transpose(0,2).numpy()).astype('double')
    explanation=explainer.explain_instance(inp,classify,top_labels=2,num_samples=1000,hide_color=0,segmentation_fn=segmenter)
    exp=explanation.local_exp[correct]
    basevec=exptovec(exp,len(exp))
    #records.append([])
    vec = 0.0
    for (sample2,_) in sampled:
        countsampled += 1
        print("At pt "+str(countpt)+" sampled "+str(countsampled))
        classify=makeclassify(sample2)
        inp=val2rgb(sample['image'].transpose(0,2).numpy()).astype('double')
        explanation=explainer.explain_instance(inp,classify,top_labels=2,num_samples=1000,hide_color=0,segmentation_fn=segmenter)
        exp=explanation.local_exp[correct]
        vec=exptovec(exp,len(exp))+ vec
        #cd = spatial.distance.cosine(vec,basevec)
        #records[-1].append(cd)
        #print(cd)
    vec /= len(sampled)
    dist=spatial.distance.euclidean(basevec,vec)
    print(dist)
    records.append(dist)

def ave(l):
    total=0.0
    for i in l:
        total += i
    return total / len(l)

print(ave(records))








