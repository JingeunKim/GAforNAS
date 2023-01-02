import math
import torch.nn as nn
import torch
import torch.nn.functional as F


def make_initial(layers, conv_unit):
    layers += [nn.Conv2d(3, int(conv_unit), kernel_size=(3, 3), stride=(1, 1)), nn.ReLU(inplace=True)]
    return layers


def conv_bat(layers, conv_unit, size_last_unit):
    layers += [nn.Conv2d(size_last_unit, int(conv_unit), kernel_size=(3, 3), padding=(1, 1)),
               nn.BatchNorm2d(int(conv_unit)), nn.ReLU(inplace=True)]
    return layers


def conv(layers, conv_unit, size_last_unit):
    layers += [nn.Conv2d(size_last_unit, int(conv_unit), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
               nn.ReLU(inplace=True)]
    return layers


def pool(layers, conv_unit):
    layers += [nn.MaxPool2d(3, int(conv_unit), 1)]
    return layers


def fclayer(layers, conv_unit, size_last_unit, dense):
    layers += [nn.Linear(size_last_unit * int(dense) * int(dense), int(conv_unit)),
               nn.ReLU(inplace=True)]
    return layers


def lastfclayer(layers, conv_unit, size_last_unit):
    layers += [nn.Linear(size_last_unit, conv_unit), nn.Softmax(dim=1)]
    return layers


def residual(layers, size_last_unit):
    layers += [nn.Conv2d(size_last_unit, size_last_unit // 4, 1),
               nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(size_last_unit // 4, size_last_unit // 4, 3, padding=1),
               nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(size_last_unit // 4, size_last_unit, 1)]
    return layers


def densenet(layers, size_last_unit, growth_rate):
    layers += [nn.BatchNorm2d(size_last_unit), nn.ReLU(),
               nn.Conv2d(size_last_unit, growth_rate * 4, kernel_size=1, stride=1,
                         padding=0, bias=False)]
    layers += [nn.BatchNorm2d(growth_rate * 4),
               nn.Conv2d(growth_rate * 4, growth_rate, kernel_size=3, stride=1,
                         padding=1, bias=False)]

    return layers


def senet(layers, size_last_unit, r):
    layers += [nn.AdaptiveAvgPool2d((1, 1))]
    layers += [nn.Linear(size_last_unit, size_last_unit // r, bias=False), nn.ReLU()]
    layers += [nn.Linear(size_last_unit // r, size_last_unit, bias=False), nn.Sigmoid()]
    return layers


class GANAS(nn.Module):
    def __init__(self, population, conv_unit):
        super(GANAS, self).__init__()
        size_last_unit = conv_unit[0]
        dense = 30
        fc_layers = []
        population = population.tolist()
        print(population)
        globals()["layers{}".format(0)] = []
        self.relu = nn.ReLU()
        layers0 = make_initial(globals()["layers{}".format(0)], conv_unit[0])
        self.idx = 0
        for i in range(1, len(population)):
            if int(population[i]) == 1:
                globals()["layers{}".format(self.idx)] = conv_bat(globals()["layers{}".format(self.idx)], conv_unit[i],
                                                                  size_last_unit)
                size_last_unit = conv_unit[i]
            elif int(population[i]) == 2:
                r = 12

                self.idx += 1

                globals()["senet_layers{}".format(self.idx)] = []
                globals()["densenet_layers{}".format(self.idx)] = []
                globals()["layers{}".format(self.idx)] = []
                globals()["skip_layers{}".format(self.idx)] = []

                globals()["senet_layers{}".format(self.idx)] = senet(globals()["senet_layers{}".format(self.idx)],
                                                                     size_last_unit, r)
                # size_last_unit = conv_unit[i]
                self.idx += 1

                globals()["senet_layers{}".format(self.idx)] = []
                globals()["densenet_layers{}".format(self.idx)] = []
                globals()["layers{}".format(self.idx)] = []
                globals()["skip_layers{}".format(self.idx)] = []

                if population[i + 1] == 2 or population[i + 1] == 4:
                    del globals()["senet_layers{}".format(self.idx)]
                    self.idx -= 1
                if population[i + 1] == 7 or population[i + 1] == 6:
                    self.idx -= 1

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

                globals()["senet_layers{}".format(self.idx)] = []
                globals()["densenet_layers{}".format(self.idx)] = []
                globals()["layers{}".format(self.idx)] = []
                globals()["skip_layers{}".format(self.idx)] = []

                globals()["skip_layers{}".format(self.idx)] = residual(globals()["skip_layers{}".format(self.idx)],
                                                                       size_last_unit)

                self.idx += 1
                globals()["senet_layers{}".format(self.idx)] = []
                globals()["densenet_layers{}".format(self.idx)] = []
                globals()["layers{}".format(self.idx)] = []
                globals()["skip_layers{}".format(self.idx)] = []
                if population[i + 1] == 6 or population[i + 1] == 4:
                    del globals()["skip_layers{}".format(self.idx)]
                    self.idx -= 1
                if population[i + 1] == 7 or population[i + 1] == 2:
                    self.idx -= 1

            elif int(population[i]) == 7:
                growth_rate = 32
                self.idx += 1
                globals()["senet_layers{}".format(self.idx)] = []
                globals()["densenet_layers{}".format(self.idx)] = []
                globals()["layers{}".format(self.idx)] = []
                globals()["skip_layers{}".format(self.idx)] = []

                globals()["densenet_layers{}".format(self.idx)] = densenet(
                    globals()["densenet_layers{}".format(self.idx)],
                    size_last_unit, growth_rate)
                self.idx += 1
                save_last = size_last_unit
                size_last_unit = growth_rate + save_last
                globals()["senet_layers{}".format(self.idx)] = []
                globals()["densenet_layers{}".format(self.idx)] = []
                globals()["layers{}".format(self.idx)] = []
                globals()["skip_layers{}".format(self.idx)] = []

                if population[i + 1] == 7 or population[i + 1] == 4:
                    del globals()["densenet_layers{}".format(self.idx)]
                    self.idx -= 1

                if population[i + 1] == 6 or population[i + 1] == 2:
                    self.idx -= 1
            else:
                break

        self.cnt = []
        if self.idx != 0:
            for o in range(self.idx + 1):
                if len(globals()["layers{}".format(o)]) != 0:
                    setattr(self, f'layers{o}', nn.Sequential(*globals()["layers{}".format(o)]))
                    self.cnt.append(1)
                elif len(globals()["skip_layers{}".format(o)]) != 0:
                    setattr(self, f'skip{o}', nn.Sequential(*globals()["skip_layers{}".format(o)]))
                    self.cnt.append(2)
                elif len(globals()["densenet_layers{}".format(o)]) != 0:
                    setattr(self, f'densenet{o}', nn.Sequential(*globals()["densenet_layers{}".format(o)]))
                    self.cnt.append(3)
                else:
                    setattr(self, f'SENet{o}', nn.Sequential(*globals()["senet_layers{}".format(o)]))
                    self.cnt.append(5)
        else:
            setattr(self, f'layers{0}', nn.Sequential(*globals()["layers{}".format(0)]))
            self.cnt.append(1)

        self.fc_layers = nn.Sequential(*fc_layers)
        self.cnt.append(4)

    def forward(self, x):
        if self.idx != 0:
            for i in range(len(self.cnt)):
                if self.cnt[i] == 1:
                    x = getattr(self, "layers{0}".format(i))(x)
                elif self.cnt[i] == 2:
                    identity = getattr(self, "skip{0}".format(i))(x)
                    x = x.clone() + identity
                    x = self.relu(x)
                elif self.cnt[i] == 3:
                    model = getattr(self, "densenet{0}".format(i))
                    # print(x.shape)
                    out = x
                    # print(model)
                    for o in range(6):
                        model_ = model[o]
                        out = model_(out)
                    # print("dense")
                    # print(out.shape)

                    x = torch.concatenate((x, out), 1)
                    # print("dense2")
                    # print(out.shape)
                elif self.cnt[i] == 5:
                    model = getattr(self, "SENet{0}".format(i))
                    # print("se")
                    # print(x.shape)
                    batch, channel, _, _ = x.size()
                    out = model[0](x)

                    out = out.view(batch, -1)
                    # print(out.shape)
                    for i in range(1, 5):
                        out = model[i](out)
                    # print(out.shape)
                    out = out.view(batch, channel, 1, 1)
                    # print(x.shape)
                    x = x * out
                else:
                    x = x.view(x.size(0), -1)
                    x = self.fc_layers(x)
        else:
            x = getattr(self, "layers{0}".format(0))(x)
            x = x.view(x.size(0), -1)
            x = self.fc_layers(x)
        return x
