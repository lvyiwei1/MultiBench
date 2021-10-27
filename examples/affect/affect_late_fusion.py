import sys
import os

sys.path.insert(1,os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from fusions.common_fusions import Concat
from datasets.affect.get_data import get_simple_processed_data
from unimodals.common_models import GRU, MLP

from training_structures.Supervised_Learning import train, test

from private_test_scripts.all_in_one import all_in_one_train

# mosi_raw.pkl, mosei_raw.pkl, sarcasm.pkl, humor.pkl
traindata, validdata, tests = \
    get_simple_processed_data('/home/paul/MultiBench/sarcasm.pkl')
    #get_simple_processed_data('/home/paul/MultiBench/mosi_raw.pkl',raw_path='/home/paul/MultiBench/mosi.hdf5')

# mosi/mosei
encoders=[GRU(35,70,last_only=True).cuda(), \
    GRU(74,150,last_only=True).cuda(),\
    GRU(300,600,last_only=True).cuda()]
head=MLP(820,400,2).cuda()

# humor/sarcasm
# encoders=[GRU(371,512,dropout=True,has_padding=True).cuda(), \
#     GRU(81,256,dropout=True,has_padding=True).cuda(),\
#     GRU(300,600,dropout=True,has_padding=True).cuda()]
# head=MLP(1368,512,1).cuda()

#all_modules = [*encoders, head]

fusion = Concat().cuda()

train(encoders, fusion, head, traindata, validdata, 100, optimtype=torch.optim.AdamW, lr=1e-4, save='sarcasm_lf_best.pt', weight_decay=0.01)


#all_in_one_train(trainprocess, all_modules)

print("Testing:")
model = torch.load('mosei_lf_best.pt').cuda()

test(model, tests, no_robust=True)



