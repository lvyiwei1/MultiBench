import sys
import os
sys.path.append(os.getcwd())
import torch
import numpy as np
from datasets.mimic.get_data import get_dataloader
import torch.nn.functional as F
trains,valid,test=get_dataloader(7,imputed_path='/home/paul/yiwei/im.pk',no_robust=True)
device='cuda:1'
#modeluni=torch.nn.Sequential(torch.load('encoder.pt'),torch.load('head.pt')).to(device)
#model=torch.load('best.pt').to(device)
for j in test:
    imginstance=j[0][0].numpy()
    print(imginstance)
    #print(imginstance.size())
    audinstance=j[1][0].numpy()
    correct=j[2][0]
    break
"""
def classifymulti(image):
    #print(np.array(images[0]).shape)
    #imgs=[torch.FloatTensor(i) for i in images]
    #audbatch=audinstance.repeat([len(imgs),1])
    with torch.no_grad():
        out=model([torch.FloatTensor(image).unsqueeze(0).to(device),torch.FloatTensor(audinstance).unsqueeze(0).to(device)])
    return F.softmax(out[0]).detach().cpu().numpy()
"""

"""
def classifyuni(image):
    with torch.no_grad():
        out=modeluni(torch.FloatTensor(image).unsqueeze(0).to(device))
    return F.softmax(out[0]).detach().cpu().numpy()
"""
def exptovec(exp,num_features):
    vals=exp[1]
    vec=np.zeros(num_features)
    for idx,v in vals:
        vec[idx]=v
    return vec


#from lime import lime_image
from private_test_scripts.lime_idea.StaticExplainer import StaticExplainer
explainer = StaticExplainer()

vecs=[]
for i in range(20):
    exec(open('examples/healthcare/mimic_add.py','r').read())
    modeladd=torch.load('add.pt').to(device)
    def classifyadd(image):
        #print(np.array(images[0]).shape)
        #imgs=[torch.FloatTensor(i) for i in images]
        #audbatch=audinstance.repeat([len(imgs),1])
        with torch.no_grad():
            out=modeladd([torch.FloatTensor(image).unsqueeze(0).to(device),torch.FloatTensor(audinstance).unsqueeze(0).to(device)])
        return F.softmax(out[0]).detach().cpu().numpy()
    explanation = explainer.explain_instance(imginstance,classifyadd,correct,100,2)
    vecs.append(exptovec(explanation,5))

print(vecs)
