import sys
import os
sys.path.append(os.getcwd())
import torch
import numpy as np
from datasets.mimic.get_data import get_dataloader
import torch.nn.functional as F
from scipy import spatial
trains,valid,test=get_dataloader(7,imputed_path='/home/paul/yiwei/im.pk',no_robust=True)
device='cuda:1'
modeluni=torch.nn.Sequential(torch.load('encoderm.pt'),torch.load('headm.pt')).to(device)
modeluni2=torch.nn.Sequential(torch.load('encodermm.pt'),torch.load('headmm.pt')).to(device)
model=torch.load('best.pt').to(device)
modeladd=torch.load('add.pt').to(device)

def sanitycheck(ser1,ser2,ser3):

    for j in test:
        imginstance=j[0][ser1].numpy()
        print(imginstance)
        #print(imginstance.size())
        audinstance=j[1][ser2].numpy()
        correct=j[2][ser3]
        print("correct: "+str(correct))
        break

    def classifymulti(image):
        #print(np.array(images[0]).shape)
        #imgs=[torch.FloatTensor(i) for i in images]
        #audbatch=audinstance.repeat([len(imgs),1])
        with torch.no_grad():
            out=model([torch.FloatTensor(image).unsqueeze(0).to(device),torch.FloatTensor(audinstance).unsqueeze(0).to(device)])
        return F.softmax(out[0]).detach().cpu().numpy()

    def classifyadd(image):
        #print(np.array(images[0]).shape)
        #imgs=[torch.FloatTensor(i) for i in images]
        #audbatch=audinstance.repeat([len(imgs),1])
        with torch.no_grad():
            out=modeladd([torch.FloatTensor(image).unsqueeze(0).to(device),torch.FloatTensor(audinstance).unsqueeze(0).to(device)])
        return F.softmax(out[0]).detach().cpu().numpy()

    def classifyuni(image):
        with torch.no_grad():
            out=modeluni(torch.FloatTensor(image).unsqueeze(0).to(device))
        return F.softmax(out[0]).detach().cpu().numpy()

    def classifyuni2(image):
        with torch.no_grad():
            out=modeluni2(torch.FloatTensor(image).unsqueeze(0).to(device))
        return F.softmax(out[0]).detach().cpu().numpy()
    def classifyuniTS(image):
        with torch.no_grad():
            out=modeluni(torch.FloatTensor(image).unsqueeze(0).to(device))
        return F.softmax(out[0]).detach().cpu().numpy()
    def classifyuniTS2(image):
        with torch.no_grad():
            out=modeluni2(torch.FloatTensor(image).unsqueeze(0).to(device))
        return F.softmax(out[0]).detach().cpu().numpy()

    def classifymultiTS(ts):
        with torch.no_grad():
            out=model([torch.FloatTensor(imginstance).unsqueeze(0).to(device),torch.FloatTensor(ts).unsqueeze(0).to(device)])
        return F.softmax(out[0]).detach().cpu().numpy()

    def classifyaddTS(ts):
        #print(np.array(images[0]).shape)
        #imgs=[torch.FloatTensor(i) for i in images]
        #audbatch=audinstance.repeat([len(imgs),1])
        with torch.no_grad():
            out=modeladd([torch.FloatTensor(imginstance).unsqueeze(0).to(device),torch.FloatTensor(ts).unsqueeze(0).to(device)])
        return F.softmax(out[0]).detach().cpu().numpy()

    def exptovec(exp,num_features):
        vals=exp[1]
        vec=np.zeros(num_features)
        for idx,v in vals:
            vec[idx]=v
        return vec


    #from lime import lime_image
    #from private_test_scripts.lime_idea.StaticExplainer import StaticExplainer
    #explainer = StaticExplainer()
    #explanation1 = explainer.explain_instance(imginstance,classifyuni,correct,100,2)
    #explanation2 = explainer.explain_instance(imginstance,classifyuni2,correct,100,2)
    #explanation2 = explainer.explain_instance(imginstance,classifyadd,correct,100,2)
    from private_test_scripts.lime_idea.TimeSeriesExplainer import CategoricalTimeSeriesExplainer

    explainer=CategoricalTimeSeriesExplainer()
    #explanation1 = explainer.explain_instance(audinstance,classifymultiTS,correct,1000,2)
    #explanation2 = explainer.explain_instance(audinstance,classifyaddTS,correct,1000,2)
    explanation3 = explainer.explain_instance(audinstance,classifyuniTS,correct,1000,2)
    explanation4 = explainer.explain_instance(audinstance,classifyuniTS2,correct,1000,2)

    uni=exptovec(explanation3,12)
    add=exptovec(explanation4,12)
    #multi=exptovec(explanation1,12)
    return uni,add
    #print("ser "+str(ser))
    print("uni-add "+str(spatial.distance.cosine(uni,add)))
    print("uni-multi "+str(spatial.distance.cosine(uni,multi)))
    print("add-multi "+str(spatial.distance.cosine(add,multi)))
    return uni,add,multi


