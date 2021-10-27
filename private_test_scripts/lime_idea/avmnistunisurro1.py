import sys
import os
sys.path.append(os.getcwd())
from training_structures.unimodal import train, test
from datasets.avmnist.get_data import get_dataloader
from unimodals.common_models import LeNet,MLP,Constant,Linear
from torch import nn
import torch

modalnum=0
traindata, validdata, testdata = get_dataloader('/home/paul/yiwei/avmnist/_MFAS/avmnist')
channels=6
#encoders=[LeNet(1,channels,3).cuda(),LeNet(1,channels,5).cuda()]
model=torch.load('avmnistadd.pt')
#model=torch.load('bestavmnist.pt')
#encoder =model.encoders[modalnum]
encoder=torch.load('encoder.pt')
head=Linear(channels*8,10).cuda()


train(encoder,head,traindata,validdata,20,optimtype=torch.optim.SGD,lr=0.001,weight_decay=0,modalnum=modalnum,head_only=True,save_encoder='encoderr.pt',save_head='headr.pt')

print("Testing:")
encoder=torch.load('encoderr.pt').cuda()
head = torch.load('headr.pt')
test(encoder,head,testdata,modalnum=modalnum)


