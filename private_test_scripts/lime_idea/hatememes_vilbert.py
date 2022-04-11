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

model_cls = registry.get_model_class("vilbert")
model = model_cls.from_pretrained("vilbert.finetuned.hateful_memes.from_cc_original").to(device)
datas=dataloader.dataset
#print(len(datas))
samplefrom = []

for i in datas:
    samplefrom.append((i,i['targets'].item()))


from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
sys.path.insert(1,os.getcwd())
from private_test_scripts.lime_idea.TimeSeriesExplainer import EmbeddingTimeSeriesExplainer
#segmenter=SegmentationAlgorithm('quickshift',kernel_size=4,max_dist=200,ratio=0.2)
explainer = EmbeddingTimeSeriesExplainer()

from private_test_scripts.lime_idea.general_method_2 import *
pt,_=sampling(samplefrom,540,2,False,totorch=False)
sampled,_=sampling(samplefrom,40,2,True,totorch=False)

#print(sampled[0][0]['image_feature_0'].size())

#runmethod2(model,explainer,1,0,pt,sampled,1000,10,[repeatingunsqueeze,cl_preprocess],segmentor=segmenter,preprocess=in_preprocess)
#runmethod2(model,explainer,0,1,pt,sampled,1000,10,[cl_preprocess,repeatingunsqueeze],segmentor=segmenter,preprocess=in_preprocess)


import copy
from mmf.common.sample import convert_batch_to_sample_list
import torch.nn.functional as F
def makeclassify(base,imagebase):
    def classify(inp):
        slist=[]
        for i in inp:
            q=torch.FloatTensor(i)
            b=copy.deepcopy(base)
            b['image_feature_0']=q
            b['image_info_0']=imagebase['image_info_0']
            b['dataset_name']='hateful_memes'
            b['dataset_type']='val'
            slist.append(b)
        samplelist=convert_batch_to_sample_list(slist)
        with torch.no_grad():
            #print(samplelist)
            out=model(to_device(samplelist,device))['scores']
        return F.softmax(out,dim=1).detach().cpu().numpy()
    return classify

countpt=0
records=[]
for (sample,correct) in pt:
    countpt += 1
    countsampled = 0
    classify=makeclassify(sample,sample)
    inp=sample['image_feature_0'].numpy().astype('double')
    explanation=explainer.explain_instance(inp,classify,correct,1000,2)[1]
    #print(explanation)[1]
    #exp=explanation.local_exp[correct]
    basevec=exptovec(explanation,len(explanation))
    #records.append([])
    vec = 0.0
    for (sample2,_) in sampled:
        countsampled += 1
        print("At pt "+str(countpt)+" sampled "+str(countsampled))
        classify=makeclassify(sample2,sample)
        inp=sample['image_feature_0'].numpy().astype('double')
        explanation=explainer.explain_instance(inp,classify,correct,1000,2)[1]
        #exp=explanation.local_exp[correct]
        vec=exptovec(explanation,len(explanation))+ vec
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

import torch
torch.save(records,'hatememeimagerecords.pt')






