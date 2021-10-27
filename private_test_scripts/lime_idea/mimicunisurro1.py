import sys
import os
sys.path.append(os.getcwd())
from training_structures.unimodal import train, test
from datasets.mimic.get_data import get_dataloader
from unimodals.common_models import LeNet,MLP,Constant,LinearWF
from torch import nn
import torch

modalnum=1
traindata, validdata, testdata = get_dataloader(7,imputed_path='/home/paul/yiwei/im.pk')
channels=6
#encoders=[LeNet(1,channels,3).cuda(),LeNet(1,channels,5).cuda()]
#model=torch.load('add.pt')
#model=torch.load('best.pt')
#encoder =model.encoders[modalnum]
encoder=torch.load('encodermm.pt')
head=LinearWF(720,2,flatten=True).cuda()


train(encoder,head,traindata,validdata,20,optimtype=torch.optim.SGD,lr=0.1,weight_decay=0.0,modalnum=modalnum,head_only=True,save_encoder='encoderm.pt',save_head='headm.pt')

print("Testing:")
encoder=torch.load('encoderm.pt').cuda()
head = torch.load('headm.pt')
test(encoder,head,testdata,modalnum=modalnum)


