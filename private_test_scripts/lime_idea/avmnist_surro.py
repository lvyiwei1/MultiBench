import sys
import os
sys.path.append(os.getcwd())
import torch
import numpy as np
from datasets.avmnist.get_data import get_dataloader
trains,valid,test=get_dataloader('/home/paul/yiwei/avmnist/_MFAS/avmnist',no_robust=True,normalize_audio=False)
device='cuda:1'
model=torch.load('avmnistadd.pt').to(device)
#model=torch.load('bestavmnist.pt').to(device)
modeluni=torch.nn.Sequential(torch.load('encoder.pt'),torch.load('head.pt')).to(device)
modeluni2=torch.nn.Sequential(torch.load('encoderr.pt'),torch.load('headr.pt')).to(device)

from skimage.color import gray2rgb,rgb2gray
import  torch.nn.functional as F

def sanitycheck(ser1,ser2,ser3):
    for j in test:
        imginstance=gray2rgb(j[0][ser2].squeeze().numpy())
        #print(imginstance.size())
        audinstance=j[1][ser1]
        correct=j[2][ser3].item()
        print(correct)
        break

    def classify(images):
        #print([i.shape for i in images])
        #print(np.array(images[0]).shape)
        imgs=[torch.FloatTensor(rgb2gray(i))/255.0 for i in images]
        audbatch=torch.stack(imgs,dim=0).unsqueeze(1).to(device)
        imgbatch=imginstance.repeat(len(imgs),1,1).unsqueeze(1).float().to(device)
        out=model([imgbatch,audbatch])
        #print(out)
        return F.softmax(out,dim=1).detach().cpu().numpy()

    def classifyuni(images):
        #print([i.shape for i in images])
        #print(np.array(images[0]).shape)
        imgs=[torch.FloatTensor(rgb2gray(i))/255.0 for i in images]
        imgbatch=torch.stack(imgs,dim=0).unsqueeze(1).to(device)
        #audbatch=audinstance.repeat(len(imgs),1,1).unsqueeze(1).float().to(device)
        out=modeluni(imgbatch)
        return F.softmax(out,dim=1).detach().cpu().numpy()
    def classifyuni2(images):
        #print([i.shape for i in images])
        #print(np.array(images[0]).shape)
        imgs=[torch.FloatTensor(rgb2gray(i))/255.0 for i in images]
        imgbatch=torch.stack(imgs,dim=0).unsqueeze(1).to(device)
        #audbatch=audinstance.repeat(len(imgs),1,1).unsqueeze(1).float().to(device)
        out=modeluni2(imgbatch)
        return F.softmax(out,dim=1).detach().cpu().numpy()
    from lime import lime_image
    from lime.wrappers.scikit_image import SegmentationAlgorithm
    segmenter=SegmentationAlgorithm('quickshift',kernel_size=1,max_dist=200,ratio=0.2)
    explainer = lime_image.LimeImageExplainer(random_state=0)
    #explanation = explainer.explain_instance(audinstance,classify,top_labels=10,hide_color=0,num_samples=10000,segmentation_fn=segmenter)
    explanation = explainer.explain_instance(imginstance,classifyuni2,top_labels=10,hide_color=0,num_samples=10000,segmentation_fn=segmenter)
    temp,mask=explanation.get_image_and_mask(explanation.top_labels[0],positive_only=False,num_features=10, hide_rest=False)
    def exptovec(exp,num_features):
        vals=exp
        vec=np.zeros(num_features)
        for idx,v in vals:
            vec[idx]=v
        return vec
    #return explanation.local_exp
    #return explanation
    exp=explanation.local_exp[correct]
    explanation2 = explainer.explain_instance(imginstance,classifyuni,top_labels=10,hide_color=0,num_samples=10000,segmentation_fn=segmenter)
    exp2=explanation2.local_exp[correct]
    return exptovec(exp,len(exp)),exptovec(exp2,len(exp2))



#print(mask)
