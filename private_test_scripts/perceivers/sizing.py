import sys
import os
sys.path.append(os.getcwd())
from perceiver_pytorch.multi_modality_perceiver import MultiModalityPerceiver, InputModality
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
"""
from datasets.mimic.get_data import get_dataloader
trains1,valid1,test1=get_dataloader(7,imputed_path='/home/paul/yiwei/im.pk',no_robust=True,batch_size=20)
from datasets.avmnist.get_data import get_dataloader
trains2,valid2,test2=get_dataloader('/home/paul/yiwei/avmnist/_MFAS/avmnist',no_robust=True,unsqueeze_channel=False)
from datasets.affect.get_data import get_dataloader
trains3,valid3,test3=get_dataloader('/home/paul/MultiBench/mosei_senti_data.pkl',raw_path='/home/paul/MultiBench/mosei.hdf5',batch_size=16,no_robust=True)
#trains4,valid4,test4=get_dataloader('/home/paul/MultiBench/mosi_raw.pkl',raw_path='/home/paul/MultiBench/mosi.hdf5',batch_size=3,no_robust=True)
"""
device='cuda:1'
static_modality=InputModality(
    name='static',
    input_channels=1,
    input_axis=1,
    num_freq_bands=6,
    max_freq=1
)
timeseries_modality=InputModality(
    name='timeseries',
    input_channels=1,
    input_axis=2,
    num_freq_bands=6,
    max_freq=1
)
colorless_image_modality=InputModality(
    name='colorlessimage',
    input_channels=1,
    input_axis=2,
    num_freq_bands=6,
    max_freq=1
)
audio_spec_modality=InputModality(
    name='audiospec',
    input_channels=1,
    input_axis=2,
    num_freq_bands=6,
    max_freq=1
)

feature1_modality=InputModality(
    name='feature1',
    input_channels=35,
    input_axis=1,
    num_freq_bands=3,
    max_freq=1
)
feature2_modality=InputModality(
    name='feature2',
    input_channels=74,
    input_axis=1,
    num_freq_bands=3,
    max_freq=1
)
feature3_modality=InputModality(
    name='feature3',
    input_channels=300,
    input_axis=1,
    num_freq_bands=3,
    max_freq=1
)
model = MultiModalityPerceiver(
    modalities=(static_modality,timeseries_modality),
    #modalities=(feature1_modality,feature2_modality,feature3_modality),
    #modalities=(static_modality,timeseries_modality,colorless_image_modality,audio_spec_modality,feature1_modality,feature2_modality,feature3_modality),
    depth=4,  # depth of net, combined with num_latent_blocks_per_layer to produce full Perceiver
    num_latents=12,
    # number of latents, or induced set points, or centroids. different papers giving it different names
    latent_dim=64,  # latent dimension
    cross_heads=1,  # number of heads for cross attention. paper said 1
    latent_heads=8,  # number of heads for latent self attention, 8
    cross_dim_head=64,
    latent_dim_head=64,
    num_classes=2,  # output number of classes
    attn_dropout=0.,
    ff_dropout=0.,
    weight_tie_layers=True,
    num_latent_blocks_per_layer=4 # Note that this parameter is 1 in the original Lucidrain implementation
    # whether to weight tie layers (optional, as indicated in the diagram)
)

#model.to_logitslist=torch.nn.ModuleList([torch.nn.Sequential(torch.nn.LayerNorm(64),torch.nn.Linear(64,2)).to(device),torch.nn.Sequential(torch.nn.LayerNorm(64),torch.nn.Linear(64,10)).to(device),torch.nn.Sequential(torch.nn.LayerNorm(64),torch.nn.Linear(64,2)).to(device)])

def getallparams(li):
    params = 0
    for module in li:
        for param in module.parameters():
            params += param.numel()
    return params

print(getallparams([model]))

#from private_test_scripts.perceivers.train_structure_multitask import train
#train(model,100,[trains1,trains2,trains3],[valid1,valid2,valid3],[test1,test2,test3],[['static','timeseries'],['colorlessimage','audiospec'],['feature1','feature2','feature3']],'private_test_scripts/perceivers/three_multitask11.pt',lr=0.0008,device=device,train_weights=[1.5,0.6,1.0],is_affect=[False,False,True],unsqueezing=[True,True,False],transpose=[False,False,False])
