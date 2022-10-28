import json
import torch
import matplotlib.pyplot as plt


with open("attr_zero_base_exp10.json", 'r') as file:
    content = [json.loads(line) for line in file.readlines()]

for i in range(len(content)):
    print(i)
    tensor = torch.tensor(content[0])
    print(tensor.size())
    print("symmetric:", torch.eq(tensor, torch.transpose(tensor, -1, -2)).all())

    tensor = torch.sum(tensor, dim=0) / 12
    s1 = torch.sum(tensor, dim=0)
    s2 = torch.sum(tensor, dim=1)
    print(tensor.size(), s1.size(), s2.size())
    plt.plot(s1, color='r')
    plt.plot(s2, color='g')
    plt.show()

    print()
