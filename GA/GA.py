import numpy as np
import torch
from matplotlib import pyplot as plt
import create_model
import dataloader
import train
from Configure import params_settings
import random

population_ = []


class Pop():
    def __init__(self):
        self.population_ = []
        self.population = params_settings.population
        self.pop_layer = params_settings.pop_layer

        self.conv = [16, 32, 64, 128, 256, 512, 1024]
        self.max_pooling = [2, 3]

    def initialization_pop(self):
        print("initialization")
        for pop in range(self.population):
            arr = []
            cnt = 0
            arr.append(1)
            while (cnt < self.pop_layer - 1):
                num = np.random.rand()
                if num < 0.25:  # conv layer +batch normalization
                    num = 1
                elif num < 0.5 and num >= 0.25:  # convlayer
                    num = 2
                elif num >= 0.5 and 0.7 > num:  # pooling layer
                    num = 3
                elif num >= 0.7 and num < 0.9:  # skip layer
                    num = 6
                else:  # dense
                    num = 4
                arr.append(num)

                if num == 4:
                    if len(arr) == self.pop_layer:
                        arr[len(arr) - 2] = 4
                        arr[len(arr) - 1] = 5
                    if len(arr) < self.pop_layer:
                        arr.append(5)
                    break

                if arr[cnt] == 3 and arr[cnt + 1] == 3:
                    ran = np.random.rand()
                    if ran > 0.5:
                        arr[cnt + 1] = 1
                    else:
                        arr[cnt + 1] = 2
                        break
                cnt += 1

            if arr[len(arr) - 1] == 1 or arr[len(arr) - 1] == 2 or arr[len(arr) - 1] == 3 or arr[
                len(arr) - 1] == 4 or arr[len(arr) - 1] == 6:
                arr[len(arr) - 2] = 4
                arr[len(arr) - 1] = 5

            if len(arr) != self.pop_layer:
                for op in range(self.pop_layer - len(arr)):
                    arr.append(0)

            if pop == 0:
                self.population_ = arr
            else:
                self.population_ = np.append(self.population_, arr, axis=0)

        self.population_ = self.population_.reshape(self.population, self.pop_layer)
        self.population_ = torch.from_numpy(self.population_)

        conv_unit = []
        for row in range(self.population):
            for col in range(self.pop_layer):
                if self.population_[row][col] == 1:
                    conv_unit.append(random.choice(self.conv))
                elif self.population_[row][col] == 2:
                    conv_unit.append(random.choice(self.conv))
                elif self.population_[row][col] == 3:
                    conv_unit.append(random.choice(self.max_pooling))
                elif self.population_[row][col] == 4:
                    conv_unit.append(np.random.randint(1000))
                elif self.population_[row][col] == 5:
                    conv_unit.append(10)
                elif self.population_[row][col] == 6:
                    conv_unit.append(random.choice(self.conv))
                else:
                    conv_unit.append(0)

        conv_unit = np.array(conv_unit)
        conv_unit = conv_unit.reshape(self.population, self.pop_layer)
        conv_unit = torch.from_numpy(conv_unit)

        return self.population_, conv_unit

    def select(self, error_rate, population_,
               conv_unit):
        sum_of_fitness = 0.0

        print("select")

        error = []
        for a in range(len(error_rate)):
            error.append(error_rate[a])

        error = list(map(int, error))

        for a in range(len(error)):
            sum_of_fitness += error[a]

        chromosome_probabilities = []

        for i in range(len(error)):
            chromosome_probabilities.append(error[i] / sum_of_fitness)

        a, b = np.random.choice(len(chromosome_probabilities), 2, p=chromosome_probabilities)
        if a == b:
            b = np.random.choice(len(chromosome_probabilities), 1, p=chromosome_probabilities)
            b = b.tolist()
            b = b[0]

        layer_child1, layer_child2, unit_child1, unit_child2 = self.crossover(population_[a], population_[b],
                                                                              conv_unit[a],
                                                                              conv_unit[b])
        return torch.tensor(layer_child1), torch.tensor(layer_child2), torch.tensor(unit_child1), torch.tensor(
            unit_child2)

    def crossover(self, layer_parent1, layer_parent2, unit_parent1, unit_parent2):
        print("crossover")
        layer_child1 = []
        layer_child2 = []

        len_layer_parent1 = self.pop_layer
        len_layer_parent2 = self.pop_layer

        for a in range(layer_parent1.size(0)):
            if layer_parent1[a] == 5:
                len_layer_parent1 = a

        for b in range(layer_parent2.size(0)):
            if layer_parent2[b] == 5:
                len_layer_parent2 = b

        if len_layer_parent1 >= len_layer_parent2:
            cross_pt = np.random.randint(0, len_layer_parent2)
        else:
            cross_pt = np.random.randint(0, len_layer_parent1)

        layer_child1.extend(layer_parent1[:cross_pt])
        layer_child1.extend(layer_parent2[cross_pt:])
        layer_child2.extend(layer_parent2[:cross_pt])
        layer_child2.extend(layer_parent1[cross_pt:])

        if len(layer_child1) < params_settings.pop_layer:
            for a in range(params_settings.pop_layer - len(layer_child1)):
                layer_child1.append(0)

        if len(layer_child2) < params_settings.pop_layer:
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

        if mutation_prob <= params_settings.mutation_prob:
            layer_child1, layer_child2, unit_child1, unit_child2 = self.mutation(layer_child1, layer_child2,
                                                                                 unit_child1, unit_child2)

        for check_pool in range(1, len(layer_child1)):
            if layer_child1[check_pool] == 3 and layer_child1[check_pool - 1] == 3:
                layer_child1.pop(check_pool)
                unit_child1.pop(check_pool)
                layer_child1.insert(check_pool, 3)
                unit_child1.insert(check_pool, 3)

        for check_pool in range(1, len(layer_child2)):
            if layer_child2[check_pool] == 3 and layer_child2[check_pool - 1] == 3:
                layer_child2.pop(check_pool)
                unit_child2.pop(check_pool)
                layer_child2.insert(check_pool, 3)
                unit_child2.insert(check_pool, 3)

        return layer_child1, layer_child2, unit_child1, unit_child2

    def del_pop(self, error_rate, population_, ConvUnit):
        print("delete")
        error = []
        for a in range(len(error_rate)):
            error.append(error_rate[a])

        error = list(map(int, error))

        population_ = population_.tolist()

        for x in range(2):
            max_error = error.index(max(error))
            population_.pop(max_error)
            ConvUnit = np.delete(ConvUnit, max_error, axis=0)
            error = np.delete(error, max_error, axis=0)
            error = error.tolist()

        return torch.tensor(population_), ConvUnit, error

    def mutation(self, layer_child1, layer_child2, unit_child1, unit_child2):
        randNum = np.random.randint(1, 4)
        len_layer_child1, len_layer_child2 = 0, 0
        for a in range(len(layer_child1)):
            if layer_child1[a] == 5:
                len_layer_child1 = a

        for b in range(len(layer_child2)):
            if layer_child2[b] == 5:
                len_layer_child2 = b

        if len_layer_child1 < 5 or len_layer_child2 < 5:
            return layer_child1, layer_child2, unit_child1, unit_child2

        if randNum == 1:
            print("mutation. add skip layer")
            one_point_ch1 = np.random.randint(2, len_layer_child1 - 2)
            one_point_ch2 = np.random.randint(2, len_layer_child2 - 2)

            ch1_numpy = np.array(layer_child1)
            if len(ch1_numpy) != params_settings.pop_layer:
                ch1_numpy = np.delete(ch1_numpy, one_point_ch1)
                np.put(ch1_numpy, one_point_ch1, 6)
            np.put(ch1_numpy, one_point_ch1, 6)
            layer_child1 = ch1_numpy.tolist()

            unit1_numpy = np.array(unit_child1)
            if len(unit1_numpy) != params_settings.pop_layer:
                unit1_numpy = np.delete(unit1_numpy, one_point_ch1)
                np.put(unit1_numpy, one_point_ch1, random.choice(self.conv))
            np.put(unit1_numpy, one_point_ch1, random.choice(self.conv))
            unit_child1 = unit1_numpy.tolist()

            ch2_numpy = np.array(layer_child2)
            if len(ch2_numpy) != params_settings.pop_layer:
                ch2_numpy = np.delete(ch2_numpy, one_point_ch2)
                np.put(ch2_numpy, one_point_ch2, 6)
            np.put(ch2_numpy, one_point_ch2, 6)
            layer_child2 = ch2_numpy.tolist()

            unit2_numpy = np.array(unit_child2)
            if len(unit2_numpy) != params_settings.pop_layer:
                unit2_numpy = np.delete(unit2_numpy, one_point_ch2)
                np.put(unit2_numpy, one_point_ch2, random.choice(self.conv))
            np.put(unit2_numpy, one_point_ch2, random.choice(self.conv))
            unit_child2 = unit2_numpy.tolist()



        elif randNum == 2:
            print("mutation. delete layer")
            one_point_ch1 = np.random.randint(2, len_layer_child1 - 2)
            one_point_ch2 = np.random.randint(2, len_layer_child2 - 2)

            ch1_numpy = np.array(layer_child1)
            layer_child1 = np.delete(ch1_numpy, one_point_ch1)
            layer_child1 = np.append(layer_child1, 0)
            layer_child1 = layer_child1.tolist()

            ch2_numpy = np.array(layer_child2)
            layer_child2 = np.delete(ch2_numpy, one_point_ch2)
            layer_child2 = np.append(layer_child2, 0)
            layer_child2 = layer_child2.tolist()

            unit1_numpy = np.array(unit_child1)
            unit_child1 = np.delete(unit1_numpy, one_point_ch1)
            unit_child1 = np.append(unit_child1, 0)
            unit_child1 = unit_child1.tolist()

            unit2_numpy = np.array(unit_child2)
            unit_child2 = np.delete(unit2_numpy, one_point_ch2)
            unit_child2 = np.append(unit_child2, 0)
            unit_child2 = unit_child2.tolist()


        else:
            print("mutation. reverse")

            ch1_first = layer_child1[:1]
            ch1_last = layer_child1[len_layer_child1 - 1:]
            ch1_body = layer_child1[1:len_layer_child1 - 1]
            ch1_body_reverse = ch1_body[::-1]
            layer_child1 = ch1_first + ch1_body_reverse + ch1_last

            un1_first = unit_child1[:1]
            un1_last = unit_child1[len_layer_child1 - 1:]
            un1_body = unit_child1[1:len_layer_child1 - 1]
            un1_body_reverse = un1_body[::-1]
            unit_child1 = un1_first + un1_body_reverse + un1_last

            ch2_first = layer_child2[:1]
            ch2_last = layer_child2[len_layer_child1 - 1:]
            ch2_body = layer_child2[1:len_layer_child1 - 1]
            ch2_body_reverse = ch2_body[::-1]
            layer_child2 = ch2_first + ch2_body_reverse + ch2_last

            un2_first = unit_child2[:1]
            un2_last = unit_child2[len_layer_child1 - 1:]
            un2_body = unit_child2[1:len_layer_child1 - 1]
            un2_body_reverse = un2_body[::-1]
            unit_child2 = un2_first + un2_body_reverse + un2_last

            if len(layer_child1) < self.pop_layer:
                lenth = self.pop_layer - len(layer_child1)
                for i in range(lenth):
                    layer_child1.append(0)
            if len(layer_child2) < self.pop_layer:
                lenth = self.pop_layer - len(layer_child2)
                for i in range(lenth):
                    layer_child2.append(0)
            if len(unit_child1) < self.pop_layer:
                lenth = self.pop_layer - len(unit_child1)
                for i in range(lenth):
                    unit_child1.append(0)
            if len(unit_child2) < self.pop_layer:
                lenth = self.pop_layer - len(unit_child2)
                for i in range(lenth):
                    unit_child2.append(0)

        return layer_child1, layer_child2, unit_child1, unit_child2

    def cat(self, population, conv_unit, layer_child1, layer_child2, unit_child1, unit_child2):
        print("concat")

        conv_unit = torch.Tensor(conv_unit)
        layer_child1 = layer_child1.view(1, self.pop_layer)
        layer_child2 = layer_child2.view(1, self.pop_layer)

        population = torch.cat([population, layer_child1], dim=0)
        population = torch.cat([population, layer_child2], dim=0)

        unit_child1 = unit_child1.view(1, self.pop_layer)
        unit_child2 = unit_child2.view(1, self.pop_layer)
        conv_unit = torch.cat([conv_unit, unit_child1], dim=0)
        conv_unit = torch.cat([conv_unit, unit_child2], dim=0)

        return population, conv_unit

    def evolve(self, device, population, conv_unit):
        generation_fitness = []
        all_error = []
        print("-" * 30)
        print("1 generation")
        for Chromosome in range(len(population)):
            model = create_model.GANAS(population[Chromosome], conv_unit[Chromosome]).to(device)
            trainloader, testloader = dataloader.data_loader()
            error_rate = train.training(model, trainloader, testloader, params_settings.epoch)
            all_error.append(error_rate)
        generation_fitness.append(min(all_error))
        print("init error")
        print(all_error)
        for ch in range(len(population) // 4):
            print("offsrting start")
            layer_child1, layer_child2, unit_child1, unit_child2 = self.select(all_error, population, conv_unit)
            population, conv_unit, all_error = self.del_pop(all_error, population, conv_unit)
            population, conv_unit = self.cat(population, conv_unit, layer_child1, layer_child2, unit_child1,
                                             unit_child2)
        # GA start
        for generation in range(1, params_settings.generations):
            print("-" * 30)
            print(generation + 1, "generation")
            for Chromosome in range(params_settings.population//2, len(population)):
                model = create_model.GANAS(population[Chromosome], conv_unit[Chromosome]).to(device)
                trainloader, testloader = dataloader.data_loader()
                error_rate = train.training(model, trainloader, testloader, params_settings.epoch)
                all_error.append(error_rate)
            generation_fitness.append(min(all_error))
            print(generation + 1, "error")
            print(all_error)
            for ch in range(len(population) // 4):
                print("offsrting start")
                layer_child1, layer_child2, unit_child1, unit_child2 = self.select(all_error, population, conv_unit)
                population, conv_unit, all_error = self.del_pop(all_error, population, conv_unit)
                population, conv_unit = self.cat(population, conv_unit, layer_child1, layer_child2, unit_child1,
                                                 unit_child2)

        final_rank = np.argsort(all_error)
        final_population = population[final_rank]
        final_conv_unit = conv_unit[final_rank]

        best_pop = final_population[0]
        best_conv = final_conv_unit[0]
        model = create_model.GANAS(best_pop, best_conv).to(device)

        torch.save(model, f'./model2.pt')
        print("end..")
        self.drawGA(generation_fitness)

    def drawGA(self, value):
        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        plt.plot(value)
        plt.show()
        plt.savefig('./model2.png')
