import sys
import os
sys.path.insert(1,os.getcwd())
from perceiver_pytorch.multi_modality_perceiver import InputModality
from private_test_scripts.perceivers.cutmid_model import MultiModalityPerceiver
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from datasets.affect.get_data import get_simple_processed_data
trains,valid,test=get_simple_processed_data('/home/paul/MultiBench/mosei_senti_data.pkl')
#from datasets.mimic.get_data import get_dataloader
#trains,valid,test=get_dataloader(7,imputed_path='/home/paul/yiwei/im.pk',no_robust=True)
#from datasets.avmnist.get_data import get_dataloader
#trains,valid,test=get_dataloader('/home/paul/yiwei/avmnist/_MFAS/avmnist',no_robust=True,unsqueeze_channel=False)
device='cuda:0'
#image_inputs = torch.rand(size=(3, 60, 60, 3), requires_grad=True).to(device)
#video_inputs = torch.rand(size=(3, 32, 10, 10, 3), requires_grad=True).to(device)
#audio_inputs = torch.rand(size=(3, 100, 1), requires_grad=True).to(device)
"""
video_modality = InputModality(
    name='video',
    input_channels=3,  # number of channels for each token of the input
    input_axis=3,  # number of axes, 3 for video)
    num_freq_bands=6,  # number of freq bands, with original value (2 * K + 1)
    max_freq=4.,  # maximum frequency, hyperparameter depending on how fine the data is
)
image_modality = InputModality(
    name='image',
    input_channels=3,  # number of channels for each token of the input
    input_axis=2,  # number of axes, 2 for images
    num_freq_bands=6,  # number of freq bands, with original value (2 * K + 1)
    max_freq=4.,  # maximum frequency, hyperparameter depending on how fine the data is
)
audio_modality = InputModality(
    name='audio',
    input_channels=1,  # number of channels for mono audio
    input_axis=1,  # number of axes, 2 for images
    num_freq_bands=6,  # number of freq bands, with original value (2 * K + 1)
    max_freq=8.,  # maximum frequency, hyperparameter depending on how fine the data is
)
"""
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
    modalities=(feature1_modality,feature2_modality,feature3_modality),
    depth=1,  # depth of net, combined with num_latent_blocks_per_layer to produce full Perceiver
    num_latents=12,
    # number of latents, or induced set points, or centroids. different papers giving it different names
    latent_dim=64,  # latent dimension
    cross_heads=1,  # number of heads for cross attention. paper said 1
    latent_heads=8,  # number of heads for latent self attention, 8
    cross_dim_head=64,
    latent_dim_head=64,
    num_classes=10,  # output number of classes
    attn_dropout=0.,
    ff_dropout=0.,
    weight_tie_layers=True,
    num_latent_blocks_per_layer=1  # Note that this parameter is 1 in the original Lucidrain implementation
    # whether to weight tie layers (optional, as indicated in the diagram)
).to(device)



from private_test_scripts.perceivers.train_structure import train
train(model,100,trains,valid,test,['feature1','feature2','feature3'],'private_test_scripts/perceivers/cutmidmosei.pt',lr=0.00001,device=device,unsqueezing=False)
