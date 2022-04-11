import sys
import os
sys.path.append(os.getcwd())
import torch
import numpy as np
#from datasets.avmnist.get_data import get_dataloader
device='cuda:0'
import  torch.nn.functional as F

def averageall(x):
    total=0.0
    count=0
    for i in x:
        #for j in i:
        total += i
        count += 1
    return total/count

def torchstack(x,leng):
    return torch.stack(x,dim=0).to(device)
def torchstackunsqueeze(x,leng):
    return torch.stack(x,dim=0).unsqueeze(1).to(device)
def identity(x):
    return x
def repeating(x,l):
    return x.repeat(l,1,1).float().to(device)
def repeatingunsqueeze(x,l):
    return x.repeat(l,1,1).unsqueeze(1).float().to(device)


def make_classifier(model,orig_values,influenced_modalnum,fns):
    def classify(inp):
        #print(inp)
        #print(inp.shape)
        inlist=[]
        for i in range(len(orig_values)):
            if i == influenced_modalnum:
                inlist.append(fns[i](inp,len(inp)))
            else:
                inlist.append(fns[i](orig_values[i],len(inp)))
        out=model(inlist)
        return F.softmax(out,dim=1).detach().cpu().numpy()
    return classify
    
def exptovec(exp,num_features):
        #print(exp)
        vals=exp
        #print(num_features)
        vec=np.zeros(num_features)
        for idx,v in vals:
            vec[idx]=v
        return vec

from scipy import spatial
def runmethod2(model,explainer,influenced_modalnum,influencing_modalnum,points_of_interest,sampled_points,lime_samples,num_classes,fns,segmentor=None,preprocess=identity,numsegs=10):
    records=[]
    segs=numsegs
    countpt = 0
    
    for pt in points_of_interest:
        countpt += 1
        countsampled=0
        #records.append([])
        #print(pt)
        correct=pt[-1].item()
        classify=make_classifier(model,pt[:-1],influenced_modalnum,fns)
        point=preprocess(pt[influenced_modalnum])
        if segmentor==None:
            exp=explainer.explain_instance(point,classify,correct,lime_samples,num_classes)
            basevec=exptovec(exp[1],segs)
        else:
            explanation=explainer.explain_instance(point,classify,top_labels=num_classes,num_samples=lime_samples,hide_color=0,segmentation_fn=segmentor)
            exp=explanation.local_exp[correct]
            segs=100
            #print(exp)
            #print(len(explanation.segments))
            basevec=exptovec(exp,segs)
        totalvec=0.0
        for sampled in sampled_points:
            countsampled += 1
            print("At pt "+str(countpt)+" sampled "+str(countsampled))
            ov = []
            for i in range(len(pt)-1):
                if i==influencing_modalnum:
                    ov.append(sampled[i])
                else:
                    ov.append(pt[i])
            classify=make_classifier(model,ov,influenced_modalnum,fns)
            
            if segmentor==None:
                exp=explainer.explain_instance(point,classify,correct,lime_samples,num_classes)    
                vec=exptovec(exp[1],segs)

            else:
                explanation=explainer.explain_instance(point,classify,top_labels=num_classes,num_samples=lime_samples,hide_color=0,segmentation_fn=segmentor)
                exp=explanation.local_exp[correct]
                #print(exp)
                vec=exptovec(exp,segs)
            #dis=spatial.distance.cosine(basevec,vec)
            #records[-1].append(dis)
            totalvec = vec + totalvec
        dis=spatial.distance.cosine(basevec,totalvec)
        records.append(dis)
    print("AVG: "+str(averageall(records)))
    return records
import random

def sampling(dataloader,num,num_classes,class_balance=False, seed=0,totorch=True):
    if class_balance and num % num_classes > 0:
        print("error: num samples must be divisible by num_classes")
        exit(1)
    classes=[]
    ret=[]
    for i in range(num_classes):
        classes.append(0)
    dataset=dataloader.dataset
    #print(dataset)
    count=len(dataset)
    #print(count)
    random.seed(seed)
    indexes=random.sample(range(count),count)
    curr=1
    total=0
    orders=[]
    while total<num:
        index = indexes[curr]
        data=dataset[index]
        #print(data[0])
        label=data[-1]
        if classes[label] < num//num_classes:
            ret.append(data)
            classes[label] +=1
            total+=1
            orders.append(index)
        curr += 1
        #print(ret[-1][0])
        
    if totorch:
        rett=[[torch.FloatTensor(j) for j in i[:-1]] for i in ret]
        for i in range(len(ret)):
            rett[i].append(torch.LongTensor([ret[i][-1]]))
            #print(rett[i][0])
        #print(ret[0][-1])
        return rett,orders
    return ret,orders

def sampletestacc(model,samples):
    corrects=0
    for j in samples:
        print(j[0].size())
        out=model([i.unsqueeze(0).cuda() for i in j[:-1]])
        if torch.argmax(out).item()==j[-1].item():
            corrects += 1
    return corrects/len(samples)







