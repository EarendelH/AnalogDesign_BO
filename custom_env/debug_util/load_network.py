import torch
from torchviz import make_dot

model = torch.load("../checkpoint_demo/checkpoint_000009/model/policy_1/model.pt")
print(f"Model Abstract: {model}")

for name, param in model.named_parameters():
    print(f"Layer: {name}")
    print(f"Size: {param.size()}")
    print(f"Values: \n{param.data}\n")
