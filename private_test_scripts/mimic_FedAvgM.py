import sys
import os
sys.path.append(os.getcwd())
from private_test_scripts.FedAvgMimic import train, test
from fusions.common_fusions import Concat
from datasets.mimic.get_data import get_dataloader
from unimodals.common_models import MLP, GRU
from torch import nn
import torch

#get dataloader for icd9 classification task 7
traindata, validdata, testdata = get_dataloader(-1, imputed_path='/home/pliang/yiwei/im.pk',train_batch_size_1=False)

#build encoders, head and fusion layer
encoders = [MLP(5, 10, 10,dropout=False).cuda(), GRU(12, 30,dropout=False).cuda()]
head = MLP(730, 40, 6, dropout=False).cuda()
fusion = Concat().cuda()

#train
train(encoders, fusion, head, traindata, validdata, 500,lr=0.004, auprc=False,save='best.pt')

#test
print("Testing: ")
model = torch.load('best.pt').cuda()
# dataset = 'mimic mortality', 'mimic 1', 'mimic 7'
test(model, testdata, dataset='mimic 7', auprc=False)
