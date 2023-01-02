import torch
import dataloader
import train
from Configure import params_settings

use_mps = torch.backends.mps.is_available()
device = torch.device("mps" if use_mps else "cpu")
print(device)
model = torch.load("./"+params_settings.model_name+".pt", map_location=device)
print(model)
trainloader, testloader = dataloader.data_loader()
error_rate = train.training(model, trainloader, testloader, params_settings.epochs)
