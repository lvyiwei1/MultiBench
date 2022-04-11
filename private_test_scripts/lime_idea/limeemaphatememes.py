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
num=100
batch_size=10
for i in datas:
    if i['targets'].item()==1:
        samplefrom.append(i)
    if len(samplefrom) == num:
        break
samples=samplefrom
labels=[]
for i in samples:
    print (' '.join(i['text']))
    a=input()
    try:
      labels.append(int(a))
    except:
      print(labels)
torch.save(labels,'lis.pt') 
exit(0)
#from lime import lime_image
#from lime.wrappers.scikit_image import SegmentationAlgorithm
sys.path.insert(1,os.getcwd())
from private_test_scripts.lime_idea.TextGroupExplainer import TextGroupExplainer
#segmenter=SegmentationAlgorithm('quickshift',kernel_size=4,max_dist=200,ratio=0.2)
explainer = TextGroupExplainer()


from tqdm import tqdm
import copy
from mmf.common.sample import convert_batch_to_sample_list

def getreses(slist):
    li = convert_batch_to_sample_list(slist)
    with torch.no_grad():
        out=model(to_device(li,device))['scores'][:,1].detach().cpu()
    return out

def getallreses(slist):
    lists=[]
    for s in slist:
        if len(lists)==0 or len(lists[-1])==batch_size:
            lists.append([])
        lists[-1].append(s)
    outs=[getreses(l) for l in tqdm(lists)]
    return torch.cat(outs,dim=0)

#storage=torch.zeros(num,num)
allcomp=[]
for i in range(num):
    for j in range(num):
        b=copy.deepcopy(samples[j])
        b['image_feature_0']=samples[i]['image_feature_0']
        b['image_info_0']=samples[i]['image_info_0']
        b['dataset_name']='hateful_memes'
        b['dataset_type']='val'
        allcomp.append(b)

storage=getallreses(allcomp).reshape(num,num)

import torch.nn.functional as F
def classify(data,j):
    allc=[]
    #print(data)
    for i in range(num):
        b=copy.deepcopy(data)
        b['image_feature_0']=samples[i]['image_feature_0']
        b['image_info_0']=samples[i]['image_info_0']
        b['dataset_name']='hateful_memes'
        b['dataset_type']='val'
        allc.append(b)
    outs=getallreses(allc)
    newstorage=copy.deepcopy(storage)
    newstorage[:,j]=outs
    avg0=torch.mean(newstorage[j])
    avg1=torch.mean(newstorage[:,j])
    avg=torch.mean(newstorage)
    uniout=avg0+avg1-avg
    multiout = newstorage[j,j]-uniout
    return np.array([uniout.item(),multiout.item()])

exps=[]
for i in range(num):
    print("Doing lime-emap on "+str(i))
    e1,e2=explainer.explain_instance(samples[i],classify,100,i)
    exps.append((e1,e2))

torch.save(exps,'hateexps.pt')






