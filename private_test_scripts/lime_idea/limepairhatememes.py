import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
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
print(datas[0])

'''samplefrom = []
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
torch.save(labels,'lis.pt')'''


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

sys.path.insert(1,os.getcwd())
from private_test_scripts.lime_idea.lime_image_text_pair import LimeImageTextPairExplainer

sample = datas[10]
tmp=copy.deepcopy(sample)
tmp['image_feature_0']=sample['image_feature_0']
tmp['image_info_0']=sample['image_info_0']
tmp['dataset_name']='hateful_memes'
tmp['dataset_type']='val'

samples = [tmp]
print(getreses(samples))

import torch.nn.functional as F

def classifier_fn(data):
    return getreses([data])

#explainer = LimeImageTextPairExplainer()
#out = explainer.explain_instance(tmp, classifier_fn, num_features=10, num_samples=100)

#img = sample['image_feature_0'].numpy()
#print(img)
#plt.imshow(img)
#plt.savefig("private_test_scripts/lime_idea/limepair_results/example.png")
