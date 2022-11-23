import torch
from torchviz import make_dot
import dataloader
import pytorch_model_summary

use_mps = torch.backends.mps.is_available()
device = torch.device("mps" if use_mps else "cpu")
print(device)
model = torch.load("./model.pt")
model = model.to(device)
trainloader, testloader = dataloader.data_loader()
for i, data in enumerate(trainloader, 0):
    inputs, labels = data
    inputs = inputs.to(device)
    labels = labels

make_dot(model(inputs), params=dict(model.named_parameters())).render("graph", format="png")

params = model.state_dict()
torch.onnx.export(model, inputs, "output.onnx")

print(pytorch_model_summary.summary(model, inputs, show_input=False))
