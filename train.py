from torch import optim
import torch
import torch.nn as nn
from torch.optim import lr_scheduler

import test

def training(model, trainloader, testloader, epochs):
    use_mps = torch.backends.mps.is_available()
    device = torch.device("mps" if use_mps else "cpu")
    criterion = nn.CrossEntropyLoss()
    curr_lr = 0.01
    optimizer = optim.SGD(model.parameters(), lr=curr_lr, momentum=0.9, weight_decay=0.0002)
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, 20, 0)
    scheduler = lr_scheduler.StepLR(optimizer, 10, 0.5)
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):

            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            for param in model.parameters():
                param.requires_grad = True
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
            scheduler.step()
        print("lr: ", optimizer.param_groups[0]['lr'])
    print('Finished Training')
    all_loss = test.test(testloader, model)
    return all_loss
