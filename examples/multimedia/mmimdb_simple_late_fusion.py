import sys
import os
sys.path.insert(1,os.getcwd())

import torch

from training_structures.Supervised_Learning import train, test
from fusions.common_fusions import Concat
from datasets.imdb.get_data import get_dataloader
from unimodals.common_models import Linear, MaxOut_MLP

filename = "best_lf.pt"
traindata, validdata, testdata = get_dataloader("multimodal_imdb.hdf5", "../video/mmimdb", vgg=True, batch_size=128,skip_process=True)

encoders=[MaxOut_MLP(512, 512, 300, linear_layer=False), MaxOut_MLP(512, 1024, 4096, 512, False)]
head= Linear(1024, 23).cuda()
fusion=Concat().cuda()

train(encoders,fusion,head,traindata,validdata,1000, early_stop=True,task="multilabel",\
    save=filename, optimtype=torch.optim.AdamW,lr=8e-3,weight_decay=0.01, objective=torch.nn.BCEWithLogitsLoss())

print("Testing:")
model=torch.load(filename).cuda()
test(model,testdata,method_name="lf",dataset="imdb",criterion=torch.nn.BCEWithLogitsLoss(),task="multilabel")
