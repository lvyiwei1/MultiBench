import sys
import os
sys.path.append(os.getcwd())
from training_structures.Supervised_Learning import train, test
from fusions.common_fusions import Concat
from datasets.mimic.get_data import get_dataloader
from unimodals.common_models import MLP, GRU
from torch import nn
import torch
from private_test_scripts.all_in_one import all_in_one_train,all_in_one_test
from memory_profiler import memory_usage
#get dataloader for icd9 classification task 7
traindata, validdata, testdata = get_dataloader(7, imputed_path='datasets/mimic/im.pk')

#build encoders, head and fusion layer
encoders = [MLP(5, 10, 10,dropout=False).cuda(), GRU(12, 30,dropout=False,batch_first=True).cuda()]
head = MLP(730, 40, 2, dropout=False).cuda()
fusion = Concat().cuda()
allmodules = [encoders[0],encoders[1],head,fusion]

#train
def trainprocess(): 
    train(encoders, fusion, head, traindata, validdata, 20, auprc=True)

all_in_one_train(trainprocess,allmodules)


#test
print("Testing: ")
model = torch.load('best.pt').cuda()
# dataset = 'mimic mortality', 'mimic 1', 'mimic 7'
test(model, testdata, dataset='mimic 7', auprc=True)
