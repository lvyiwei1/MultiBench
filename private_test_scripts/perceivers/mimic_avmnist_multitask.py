import sys
import os
sys.path.append(os.getcwd())
from perceiver_pytorch.multi_modality_perceiver import MultiModalityPerceiver, InputModality
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from datasets.mimic.get_data import get_dataloader
trains1,valid1,test1=get_dataloader(7,imputed_path='/home/paul/yiwei/im.pk',no_robust=True,batch_size=20)
from datasets.avmnist.get_data import get_dataloader
trains2,valid2,test2=get_dataloader('/home/paul/yiwei/avmnist/_MFAS/avmnist',no_robust=True,unsqueeze_channel=False)
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
model = MultiModalityPerceiver(
    modalities=(static_modality,timeseries_modality,colorless_image_modality,audio_spec_modality),
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
    num_latent_blocks_per_layer=4  # Note that this parameter is 1 in the original Lucidrain implementation
    # whether to weight tie layers (optional, as indicated in the diagram)
).to(device)

model.to_logitslist=torch.nn.ModuleList([torch.nn.Sequential(torch.nn.LayerNorm(64),torch.nn.Linear(64,2)).to(device),torch.nn.Sequential(torch.nn.LayerNorm(64),torch.nn.Linear(64,10)).to(device)])

from private_test_scripts.perceivers.train_structure_multitask import train
train(model,100,[trains1,trains2],[valid1,valid2],[test1,test2],[['static','timeseries'],['colorlessimage','audiospec']],'private_test_scripts/perceivers/mimic_avmnist_multitask.pt',lr=0.0005,device=device,train_weights=[1.0,0.6])
