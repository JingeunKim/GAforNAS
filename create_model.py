import math
import torch.nn as nn


def make_initial(layers, conv_unit):
    layers += [nn.Conv2d(3, int(conv_unit), 3), nn.ReLU(inplace=True)]
    return layers


def conv_bat(layers, conv_unit, size_last_unit):
    layers += [nn.Conv2d(size_last_unit, int(conv_unit), 3, padding=(1, 1)),
               nn.BatchNorm2d(int(conv_unit)), nn.ReLU(inplace=True)]
    return layers


def conv(layers, conv_unit, size_last_unit):
    layers += [nn.Conv2d(size_last_unit, int(conv_unit), 3, padding=(1, 1)), nn.ReLU(inplace=True)]
    return layers


def pool(layers, conv_unit):
    layers += [nn.MaxPool2d(3, int(conv_unit), 1), nn.ReLU(inplace=True)]
    return layers


def fclayer(layers, conv_unit, size_last_unit, dense):
    layers += [nn.Linear(size_last_unit * int(dense) * int(dense), int(conv_unit)),
               nn.ReLU(inplace=True)]
    return layers


def lastfclayer(layers, conv_unit, size_last_unit):
    layers += [nn.Linear(size_last_unit, conv_unit), nn.Softmax(dim=1)]
    return layers


def residual(layers, size_last_unit):
    layers += [nn.Conv2d(size_last_unit, size_last_unit, 3, padding=1),
               nn.BatchNorm2d(size_last_unit), nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(size_last_unit, size_last_unit, 3, padding=1),
               nn.BatchNorm2d(size_last_unit), nn.ReLU(inplace=True)]
    return layers


class GANAS(nn.Module):
    def __init__(self, population, conv_unit):
        super(GANAS, self).__init__()
        size_last_unit = conv_unit[0]
        dense = 30
        fc_layers = []
        population = population.tolist()
        num_six = population.count(6)

        print(population)

        globals()["layers{}".format(0)] = []

        layers0 = make_initial(globals()["layers{}".format(0)], conv_unit[0])

        self.idx = 0
        for i in range(1, len(population)):
            if int(population[i]) == 1:
                globals()["layers{}".format(self.idx)] = conv_bat(globals()["layers{}".format(self.idx)], conv_unit[i],
                                                                  size_last_unit)
                size_last_unit = conv_unit[i]
            elif int(population[i]) == 2:
                globals()["layers{}".format(self.idx)] = conv(globals()["layers{}".format(self.idx)], conv_unit[i],
                                                              size_last_unit)
                size_last_unit = conv_unit[i]
            elif int(population[i]) == 3:
                globals()["layers{}".format(self.idx)] = pool(globals()["layers{}".format(self.idx)], conv_unit[i])
                if int(conv_unit[i]) == 2:
                    dense = math.ceil(dense / 2)
                else:
                    dense = math.ceil(dense / 3)
            elif int(population[i]) == 4:
                if dense <= 0:
                    dense = 1
                fc_layers = fclayer(fc_layers, conv_unit[i], size_last_unit, dense)
                size_last_unit = conv_unit[i]
            elif int(population[i]) == 5:
                fc_layers = lastfclayer(fc_layers, conv_unit[i], size_last_unit)
            elif int(population[i]) == 6:
                self.idx += 1
                globals()["skip_layers{}".format(self.idx)] = []
                globals()["layers{}".format(self.idx)] = []
                globals()["skip_layers{}".format(self.idx)] = residual(globals()["skip_layers{}".format(self.idx)],
                                                                       size_last_unit)

                self.idx += 1
                globals()["layers{}".format(self.idx)] = []
                globals()["skip_layers{}".format(self.idx)] = []
                if population[i + 1] == 6 or population[i + 1] == 4:
                    del globals()["skip_layers{}".format(self.idx)]
                    self.idx -= 1
            else:
                break

        self.cnt = []
        if self.idx != 0:
            for o in range(self.idx + 1):
                if len(globals()["layers{}".format(o)]) != 0:
                    setattr(self, f'layers{o}', nn.Sequential(*globals()["layers{}".format(o)]))
                    self.cnt.append(1)
                else:
                    setattr(self, f'skip{o}', nn.Sequential(*globals()["skip_layers{}".format(o)]))
                    self.cnt.append(2)
        else:
            setattr(self, f'layers{0}', nn.Sequential(*globals()["layers{}".format(0)]))
            self.cnt.append(1)

        self.fc_layers = nn.Sequential(*fc_layers)
        self.cnt.append(3)

    def forward(self, x):
        if self.idx != 0:
            for i in range(len(self.cnt)):
                if self.cnt[i] == 1:
                    x = getattr(self, "layers{0}".format(i))(x)
                elif self.cnt[i] == 2:
                    identity = getattr(self, "skip{0}".format(i))(x)
                    x = x.clone() + identity
                elif self.cnt[i] == 3:
                    x = x.view(x.size(0), -1)
                    x = self.fc_layers(x)
        else:
            x = getattr(self, "layers{0}".format(0))(x)
            x = x.view(x.size(0), -1)
            x = self.fc_layers(x)
        return x
