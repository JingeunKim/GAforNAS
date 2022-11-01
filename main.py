import torch  # conda install pytorch torchvision torchaudio -c pytorch-nightly
import numpy as np
from GA import GA
import gc
from torch import optim
import create_model
import train
import dataloader
from Configure import params_settings

# 파이토치,mps 버전 확인
print("Torch version:{}".format(torch.__version__))
print(f"mps 사용 가능 여부: {torch.backends.mps.is_available()}")
print(f"mps 지원 환경 여부: {torch.backends.mps.is_built()}")
use_mps = torch.backends.mps.is_available()
device = torch.device('mps' if use_mps else 'cpu')
print(device)

# 초기 인구
pop = GA.Pop()
population, conv_unit = pop.initialization_pop()


for generation in range(params_settings.generations):
    print("-"*30)
    print(generation+1, "STart")
    all_error = []
    print(population, conv_unit)
    for Chromosome in range(len(population)):
        layers, num_dense, last_conv_num = pop.make_cp(population[Chromosome], conv_unit[Chromosome])
        fclayers = pop.make_fc(population[Chromosome], conv_unit[Chromosome], num_dense, last_conv_num)

        model = create_model.GANAS(layers, fclayers).to(device)
        print(model)

        trainloader, testloader = dataloader.data_loader()
        error_rate = train.training(model, trainloader, testloader)
        all_error.append(error_rate)
    print("all_loss = ", all_error)
    for ch in range(len(population) // 2):
        print("-"*30)
        print("offsrting start")
        layer_child1, layer_child2, unit_child1, unit_child2 = pop.select(all_error, population, conv_unit)
        print("offspring", layer_child1, layer_child2, unit_child1, unit_child2)
        population, conv_unit = pop.del_pop(all_error, population, conv_unit)  # 순위 낮은 염색체 삭제
        print("삭제 확인", population, conv_unit, type(population), type(conv_unit), type(layer_child1), type(layer_child2),
              type(unit_child1), type(unit_child2))
        # layer_child1, layer_child2, unit_child1, unit_child2 = pop.mutation(layer_child1, layer_child2, unit_child1, unit_child2) # 돌연변이 실행
        population, conv_unit = pop.cat(population, conv_unit, layer_child1, layer_child2, unit_child1,
                                        unit_child2)  # 자식 추가
        print("자식 추가 확인", population, conv_unit)
