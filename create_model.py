import torch.nn as nn

class GANAS(nn.Module):  # Conv 는 -2, Maxpooling 은 /2 flatten 하는 레이어는 마지막 conv아웃풋 레이어 * (32 - Conv)
    def __init__(self, layers, fclayers):
        super(GANAS, self).__init__()
        self.layers = nn.Sequential(*layers)
        self.fclayers = fclayers

    def forward(self, x):
        x = self.layers(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.fclayers(x)

        return x