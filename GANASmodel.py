import torch
import dataloader
import train
from Configure import params_settings

use_mps = torch.backends.mps.is_available()
device = torch.device("mps" if use_mps else "cpu")
print(device)
model = torch.load("./model2.pt", map_location=device)
trainloader, testloader = dataloader.data_loader()
error_rate = train.training(model, trainloader, testloader, params_settings.epochs)
print(error_rate)