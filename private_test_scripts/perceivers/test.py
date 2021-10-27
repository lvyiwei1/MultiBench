import sys
import os
sys.path.append(os.getcwd())
from perceiver_pytorch.multi_modality_perceiver import MultiModalityPerceiver, InputModality
import torch
device="cuda:1"

# Import your dataset like this
from datasets.mimic.get_data import get_dataloader
trains,valid,test=get_dataloader(7,imputed_path='/home/pliang/yiwei/im.pk',no_robust=True)

# Define your modalities like the following examples
video_modality = InputModality(
    name='video',
    input_channels=3,  # number of channels for each token of the input
    input_axis=3,  # number of axes, 3 for video)
    num_freq_bands=6,  # number of freq bands, with original value (2 * K + 1)
    max_freq=4.,  # maximum frequency, hyperparameter depending on how fine the data is
) # the input for this modality will need to be of shape (batch_size,anything,anything,anything,3)

image_modality = InputModality(
    name='image',
    input_channels=3,  # number of channels for each token of the input
    input_axis=2,  # number of axes, 2 for images
    num_freq_bands=6,  # number of freq bands, with original value (2 * K + 1)
    max_freq=4.,  # maximum frequency, hyperparameter depending on how fine the data is
)  # the input for this modality will need to be of shape (batch_size,anything,anything,3)

audio_modality = InputModality(
    name='audio',
    input_channels=1,  # number of channels for mono audio
    input_axis=1,  # number of axes, 2 for images
    num_freq_bands=6,  # number of freq bands, with original value (2 * K + 1)
    max_freq=8.,  # maximum frequency, hyperparameter depending on how fine the data is
) # the input for this modality will need to be of shape (batch_size,anything,1)

static_modality=InputModality(
    name='static',
    input_channels=1,
    input_axis=1,
    num_freq_bands=6,
    max_freq=1
) # for mimic static modality,  the input for this modality will need to be of shape (batch_size,anything,1)

timeseries_modality=InputModality(
    name='timeseries',
    input_channels=1,
    input_axis=2,
    num_freq_bands=6,
    max_freq=1
) # for mimic time series modality, the input for this modality will need to be of shape (batch_size,anything,anything,1)

verylarge_modality=InputModality(
    name='verylarge',
    input_channels=3,
    input_axis=3,
    num_freq_bands=6,
    max_freq=4
) # this modality is used to make the max dimension large enough so that the perceivers are compatible across different datasets

# use the following code to create new model with untrained parameters. You may change the hyperparameters to reduce complexity
#"""
model = MultiModalityPerceiver(
    modalities=(static_modality,timeseries_modality,verylarge_modality),
    depth=8,  # depth of net, combined with num_latent_blocks_per_layer to produce full Perceiver
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
    num_latent_blocks_per_layer=6  # Note that this parameter is 1 in the original Lucidrain implementation
    # whether to weight tie layers (optional, as indicated in the diagram)
).to(device)
#"""

# Alternatively, load a already-trained perceiver like following
#model=torch.load("private_test_scripts/perceivers/avmnist.pt").to(device)# load whatever pretrained perceiver checkpoint you want to start with
#model.modalities={'static':static_modality,'timeseries':timeseries_modality} # you will need to change this to the specific modalities in your dataset
#model.to_logits=torch.nn.Sequential(torch.nn.LayerNorm(64),torch.nn.Linear(64,2)).to(device) # change 64 to your latent_dim and change 2 to the number of classes in your classification task, if necessary

# train the perceiver. Change the modalities parameter to a list of input modalities of your dataset, and change the savedir parameter to whereever you want to store the best validation performing checkpoint
from private_test_scripts.perceivers.train_structure import train
train(model,100,trains,valid,test,['static','timeseries'],'private_test_scripts/perceivers/test.pt',lr=0.001,device=device)
