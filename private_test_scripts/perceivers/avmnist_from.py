import sys
import os
sys.path.append(os.getcwd())
from perceiver_pytorch.multi_modality_perceiver import MultiModalityPerceiver, InputModality
import torch
from datasets.avmnist.get_data import get_dataloader
trains,valid,test=get_dataloader('/home/pliang/yiwei/avmnist/_MFAS/avmnist',no_robust=True,unsqueeze_channel=False)
#image_inputs = torch.rand(size=(3, 60, 60, 3), requires_grad=True).cuda()
#video_inputs = torch.rand(size=(3, 32, 10, 10, 3), requires_grad=True).cuda()
#audio_inputs = torch.rand(size=(3, 100, 1), requires_grad=True).cuda()
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
"""
model = MultiModalityPerceiver(
    modalities=(colorless_image_modality,audio_spec_modality),
    depth=8,  # depth of net, combined with num_latent_blocks_per_layer to produce full Perceiver
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
    num_latent_blocks_per_layer=6  # Note that this parameter is 1 in the original Lucidrain implementation
    # whether to weight tie layers (optional, as indicated in the diagram)
).cuda()
"""
model=torch.load("private_test_scripts/perceivers/mimic.pt").cuda()
model.modalities={'colorlessimage':colorless_image_modality,'audiospec':audio_spec_modality}
model.to_logits=torch.nn.Sequential(torch.nn.LayerNorm(64),torch.nn.Linear(64,10)).cuda()

from private_test_scripts.perceivers.train_structure import train
train(model,50,trains,valid,test,['colorlessimage','audiospec'],'private_test_scripts/perceivers/avmnist_from.pt',lr=0.001)
