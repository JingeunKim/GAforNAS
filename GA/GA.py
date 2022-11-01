import numpy as np
import torch
from Configure import params_settings
import torch.nn as nn
import math

population_ = []


class Pop():
    def __init__(self):
        self.population_ = []
        self.population = params_settings.population
        self.pop_layer = params_settings.pop_layer

    def initialization_pop(self):
        for pop in range(self.population):
            arr = []
            cnt = 0
            arr.append(1)
            while (cnt < self.pop_layer - 1):  ## 맨앞은 무조건 conv, pooling 2번 불가능, dense나오면 하나더 추가하고 끝내기
                num = np.random.rand()
                if num < 0.5:  # conv는 50%, max는 45%, dense는 5%의 확률로 생성됨
                    num = 1
                elif num >= 0.5 and 0.95 > num:
                    num = 2
                else:
                    num = 3
                arr.append(num)

                if num == 3:  # FC를 받으면 FC를 하나 더 추가한 후 종료
                    if len(arr) == self.pop_layer:
                        arr[len(arr) - 2] = 3
                        arr[len(arr) - 1] = 4
                    if len(arr) < self.pop_layer:
                        arr.append(4)
                    break

                if arr[cnt] == 2 and arr[cnt + 1] == 2:  # max_pooling이 연속 2번나오는 것을 방지
                    ran = np.random.rand()
                    if ran > 0.5:
                        arr[cnt + 1] = 1
                    else:
                        arr[cnt + 1] = 3
                        if len(arr) < self.pop_layer:
                            arr.append(4)
                        break
                cnt += 1

            if arr[len(arr) - 1] == 1 or arr[len(arr) - 1] == 2 or arr[
                len(arr) - 1] == 3:  # 배열의 끝이 conv or max_pooling일 시 마지막에서 두번째 & 마지막을 fc로 변환
                arr[len(arr) - 2] = 3
                arr[len(arr) - 1] = 4

            if len(arr) != self.pop_layer:
                for op in range(self.pop_layer - len(arr)):
                    arr.append(0)

            if pop == 0:
                self.population_ = arr
            else:
                self.population_ = np.append(self.population_, arr, axis=0)

        self.population_ = self.population_.reshape(self.population, self.pop_layer)
        self.population_ = torch.from_numpy(self.population_)

        import random

        conv = [16, 32, 64, 128, 256, 512, 1024]
        max_pooling = [2, 3]  # kernel
        dense = np.random.randint(1000)

        conv_unit = []
        for row in range(self.population):
            for col in range(self.pop_layer):
                if self.population_[row][col] == 1:
                    conv_unit.append(random.choice(conv))
                elif self.population_[row][col] == 2:
                    conv_unit.append(random.choice(max_pooling))
                elif self.population_[row][col] == 3:
                    conv_unit.append(np.random.randint(1000))
                elif self.population_[row][col] == 4:
                    conv_unit.append(10)
                else:
                    conv_unit.append(0)

        conv_unit = np.array(conv_unit)
        conv_unit = conv_unit.reshape(self.population, self.pop_layer)

        # print(self.population_)
        # print("-" * 80)
        # print(conv_unit)
        return self.population_, conv_unit

    def make_cp(self, layerArray, UnitArray):
        layers = []
        layer_cnt = []
        layers += [nn.Conv2d(3, int(UnitArray[0]), 3), nn.ReLU(inplace=True)]
        Conv_last_layer = int(UnitArray[0])
        layer_cnt.append(1)
        dense = 30
        for i in range(1, len(layerArray)):
            if layerArray[i] == 1:
                add_bn_prob = np.random.rand()
                if add_bn_prob <= 0.5:
                    layers += [nn.Conv2d(Conv_last_layer, int(UnitArray[i]), 3, padding=(1, 1)),
                               nn.BatchNorm2d(int(UnitArray[i])), nn.ReLU(inplace=True)]
                else:
                    layers += [nn.Conv2d(Conv_last_layer, int(UnitArray[i]), 3, padding=(1, 1)), nn.ReLU(inplace=True)]
                Conv_last_layer = int(UnitArray[i])
                layer_cnt.append(1)

            elif layerArray[i] == 2:
                layers += [nn.MaxPool2d(3, int(UnitArray[i]), 1), nn.ReLU(inplace=True)]

                if int(UnitArray[i]) == 2:
                    dense = math.ceil(dense / 2)
                else:
                    dense = math.ceil(dense / 3)
                layer_cnt.append(2)
        if dense <= 0:
            dense = 1
        # print("aaa",dense)
        return nn.Sequential(*layers), dense, Conv_last_layer

    def make_fc(self, layerArray, UnitArray, dense, Conv_last_layer):
        layers = []
        layer_cnt = []

        for i in range(1, len(layerArray)):
            if layerArray[i] == 3:
                layers += [nn.Linear(Conv_last_layer * int(dense) * int(dense), int(UnitArray[i])),
                           nn.ReLU(inplace=True)]
                layer_cnt.append(3)
            elif layerArray[i] == 4:
                layers += [nn.Linear(int(UnitArray[i - 1]), 10), nn.Softmax(dim=1)]
                layer_cnt.append(4)

        return nn.Sequential(*layers)

    def select(self, error_rate, population_,
               conv_unit):  # (all_error, params_settings.population, population, conv_unit)
        sum_of_fitness = 0.0

        error = []
        for a in range(len(error_rate)):
            error.extend(error_rate[a])

        error = list(map(int, error))

        for a in range(len(error)):
            sum_of_fitness += error[a]

        chromosome_probabilities = []
        print("all loss = ", sum_of_fitness)
        for i in range(params_settings.population):
            chromosome_probabilities.append(error[i] / sum_of_fitness)

        print("가능성의 합은 = ", sum(chromosome_probabilities))
        a, b = np.random.choice(params_settings.population, 2, p=chromosome_probabilities)
        print("지목된 부모 인덱스 = ", a, b)
        print("지목된 배열 두개 = ", population_[a], "  ", population_[b])

        layer_child1, layer_child2, unit_child1, unit_child2 = self.crossover(population_[a], population_[b],
                                                                              conv_unit[a],
                                                                              conv_unit[b])
        return torch.tensor(layer_child1), torch.tensor(layer_child2), torch.tensor(unit_child1), torch.tensor(
            unit_child2)

    def crossover(self, layer_parent1, layer_parent2, unit_parent1, unit_parent2):
        layer_child1 = []
        layer_child2 = []

        # print(layer_parent1, layer_parent2, unit_parent1, unit_parent2)
        len_layer_parent1 = self.pop_layer
        len_layer_parent2 = self.pop_layer
        for a in range(layer_parent1.size(0)):
            if layer_parent1[a] == 4:
                len_layer_parent1 = a

        for b in range(layer_parent2.size(0)):
            if layer_parent1[b] == 4:
                len_layer_parent2 = b

        # print(a, b)
        print("일단 길이 = ", len_layer_parent1, len_layer_parent2)
        if len_layer_parent1 >= len_layer_parent2:  # 길이가 짧은 쪽으로 포인트 정함
            cross_pt = np.random.randint(0, len_layer_parent2)
        else:
            cross_pt = np.random.randint(0, len_layer_parent1)

        print("cross pt = ", cross_pt)
        layer_child1.extend(layer_parent1[:cross_pt])
        print("layer_parent1[:cross_pt] = ", layer_parent1[:cross_pt])
        layer_child1.extend(layer_parent2[cross_pt:])
        print("layer_parent2[cross_pt:] = ", layer_parent2[cross_pt:])
        print(layer_child1)
        print("----------자식교배 1------")
        layer_child2.extend(layer_parent2[:cross_pt])
        print("layer_parent2[:cross_pt] = ", layer_parent2[:cross_pt])
        layer_child2.extend(layer_parent1[cross_pt:])
        print("layer_parent1[:cross_pt] = ", layer_parent1[cross_pt:])
        print(layer_child2)
        print("----------자식교배 2------")

        if len(layer_child1) != params_settings.pop_layer:
            for a in range(params_settings.pop_layer - len(layer_child1)):
                layer_child1.append(0)

        if len(layer_child2) != params_settings.pop_layer:
            for a in range(params_settings.pop_layer - len(layer_child2)):
                layer_child2.append(0)

        unit_child1 = []
        unit_child2 = []

        unit_child1.extend(unit_parent1[:cross_pt])
        unit_child1.extend(unit_parent2[cross_pt:])
        unit_child2.extend(unit_parent2[:cross_pt])
        unit_child2.extend(unit_parent1[cross_pt:])

        if len(unit_child1) != params_settings.pop_layer:
            for a in range(params_settings.pop_layer - len(unit_child1)):
                unit_child1.append(0)

        if len(unit_child2) != params_settings.pop_layer:
            for a in range(params_settings.pop_layer - len(unit_child2)):
                unit_child2.append(0)

        mutation_prob = np.random.rand()
        if mutation_prob <= 0.9:  # 돌연변이 확률
            print("완전 돌연변이 가기 전 = ", layer_child1, layer_child2, unit_child1, unit_child2)
            layer_child1, layer_child2, unit_child1, unit_child2 = self.mutation(layer_child1, layer_child2,
                                                                                 unit_child1, unit_child2)

        print("muation 끝난 후", layer_child1, layer_child2, unit_child1, unit_child2)
        print(type(layer_child1))
        # pool layer 연속으로 나오는 것을 방지
        print("ch1 len = ", len(layer_child1))
        print("ch2 len = ", len(layer_child2))
        for check_pool in range(1, len(layer_child1)):
            print("check_pool = ", check_pool)
            if layer_child1[check_pool] == 2 and layer_child1[check_pool - 1] == 2:
                layer_child1.pop(check_pool)
                unit_child1.pop(check_pool)

        for check_pool in range(1, len(layer_child2)):
            print("check_pool = ", check_pool)
            if layer_child2[check_pool] == 2 and layer_child2[check_pool - 1] == 2:
                layer_child2.pop(check_pool)
                unit_child2.pop(check_pool)

        return layer_child1, layer_child2, unit_child1, unit_child2

    def del_pop(self, error_rate, population_, ConvUnit):
        error = []
        for a in range(len(error_rate)):
            error.extend(error_rate[a])

        error = list(map(int, error))

        low_list = np.argsort(error)
        print(low_list)
        population_ = population_.tolist()
        print(population_)

        for i in range(2):  # 부모가 자식 2명을 만들기 때문에 기존 인구 중 2명 삭제
            population_.pop(low_list[i])
            ConvUnit = np.delete(ConvUnit, low_list[i], axis=0)
            low_list -= 1
        return torch.tensor(population_), ConvUnit

    def mutation(self, layer_child1, layer_child2, unit_child1, unit_child2):
        print("muation 전 ", layer_child1, layer_child2, unit_child1, unit_child2)
        randNum = np.random.randint(1, 4)
        print("mutation random num = ", randNum)
        len_layer_child1, len_layer_child2 = 0, 0
        for a in range(len(layer_child1)):
            if layer_child1[a] == 4:
                len_layer_child1 = a

        for b in range(len(layer_child2)):
            if layer_child2[b] == 4:
                len_layer_child2 = b

        if len_layer_child1 < 5 or len_layer_child2 < 5:  # 둘중 하나라도 길이가 5를 넘지 못하면 돌연변이 만들지 않음
            return layer_child1, layer_child2, unit_child1, unit_child2

        print("레이어 길이 확인", len_layer_child1, len_layer_child2)
        print("pre mutation = ", layer_child1, layer_child2, unit_child1, unit_child2)
        if randNum == 1:  # residual block
            pass
        elif randNum == 2:  # delete conv or pool layerv

            one_point_ch1 = np.random.randint(2, len_layer_child1 - 2)  # 자식1 삭제할 점
            one_point_ch2 = np.random.randint(2, len_layer_child2 - 2)  # 자식2 삭제할 점
            print("2시작")
            print("포인트 = ", one_point_ch1, one_point_ch2)
            # child1의 layer 돌연변이
            print("case2 pre layer_child1 = ", layer_child1)
            ch1_numpy = np.array(layer_child1)
            layer_child1 = np.delete(ch1_numpy, one_point_ch1)
            print("case2 pre2 layer_child1 = ", layer_child1)
            layer_child1 = np.append(layer_child1, 0)
            print("case2 final layer_child1 = ", layer_child1)
            layer_child1 = layer_child1.tolist()
            # layer_child1 = torch.Tensor(ch1_numpy)
            # child2의 layer 돌연변이
            ch2_numpy = np.array(layer_child2)
            layer_child2 = np.delete(ch2_numpy, one_point_ch2)
            layer_child2 = np.append(layer_child2, 0)
            layer_child2 = layer_child2.tolist()
            # layer_child2 = torch.Tensor(ch2_numpy)
            # child1의 unit 돌연변이
            unit1_numpy = np.array(unit_child1)
            unit_child1 = np.delete(unit1_numpy, one_point_ch1)
            unit_child1 = np.append(unit_child1, 0)
            unit_child1 = unit_child1.tolist()
            # unit_child1 = torch.Tensor(unit1_numpy)
            # child2의 unit 돌연변이
            unit2_numpy = np.array(unit_child2)
            unit_child2 = np.delete(unit2_numpy, one_point_ch2)
            unit_child2 = np.append(unit_child2, 0)
            unit_child2 = unit_child2.tolist()
            # unit_child2 = torch.Tensor(unit2_numpy)
            print("muation2 ", layer_child1, layer_child2, unit_child1, unit_child2)

        else:  # change conv layer <--> pool layer
            one_point_ch1 = np.random.randint(2, len_layer_child1 - 2)  # 자식1 레이어 교환 점
            one_point_ch2 = np.random.randint(2, len_layer_child2 - 2)  # 자식2 레이어 교환 점
            print("case3 pre ex =", layer_child1)
            print("p1 = ", one_point_ch1, " p2 = ", one_point_ch2)
            # 자손1
            ch1_first = layer_child1[2:one_point_ch1]
            ch1_second = layer_child1[one_point_ch1:len_layer_child1 - 2]
            print("muation3.1 ", ch1_first, ch1_second)

            layer_child1[2:one_point_ch1 + 2] = ch1_second
            layer_child1[one_point_ch1 + 2:len_layer_child1 - 2] = ch1_first
            print("muation3.2 ", layer_child1)

            unit1_first = unit_child1[2:one_point_ch1]
            unit1_second = unit_child1[one_point_ch1:len_layer_child1 - 2]

            unit_child1[2:one_point_ch1 + 2] = unit1_second
            unit_child1[one_point_ch1 + 2:len_layer_child1 - 2] = unit1_first

            # 자손2
            ch2_first = layer_child2[2:one_point_ch2]
            ch2_second = layer_child2[one_point_ch2:len_layer_child2 - 2]

            layer_child2[2:one_point_ch2 + 2] = ch2_second
            layer_child2[one_point_ch2 + 2:len_layer_child2 - 2] = ch2_first

            unit2_first = unit_child2[2:one_point_ch2]
            unit2_second = unit_child2[one_point_ch2:len_layer_child2 - 2]

            unit_child2[2:one_point_ch2 + 2] = unit2_second
            unit_child2[one_point_ch2 + 2:len_layer_child2 - 2] = unit2_first
            print("muation3 ", layer_child1, layer_child2, unit_child1, unit_child2)

        return layer_child1, layer_child2, unit_child1, unit_child2

    def cat(self, population, conv_unit, layer_child1, layer_child2, unit_child1, unit_child2):
        print("concat시작")
        conv_unit = torch.Tensor(conv_unit)
        print("origin population = ", population)
        print("origin conv_unit = ", conv_unit)
        layer_child1 = layer_child1.view(1, self.pop_layer)
        layer_child2 = layer_child2.view(1, self.pop_layer)
        print("origin layer_child1 = ", layer_child1)
        print("origin layer_child2 = ", layer_child2)

        population = torch.cat([population, layer_child1], dim=0)
        population = torch.cat([population, layer_child2], dim=0)
        print("post population = ", population)

        unit_child1 = unit_child1.view(1, self.pop_layer)
        unit_child2 = unit_child2.view(1, self.pop_layer)
        print("origin unit_child2 = ", unit_child2)
        print("origin unit_child2 = ", unit_child2)

        conv_unit = torch.cat([conv_unit, unit_child1], dim=0)
        conv_unit = torch.cat([conv_unit, unit_child2], dim=0)
        print("post conv_unit = ", conv_unit)

        return population, conv_unit
