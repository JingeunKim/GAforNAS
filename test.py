import torch


def test(testloader, model):
    all_loss = []
    correct = 0
    total = 0
    print("test")
    with torch.no_grad():
        use_mps = torch.backends.mps.is_available()
        device = torch.device("mps" if use_mps else "cpu")
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            acc = 100 * correct // total
            error_rate = 100 - acc

    print(f'Error rate of the network on the 10000 test images: {error_rate} %')
    all_loss.append(error_rate)
    return all_loss
