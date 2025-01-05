import torch
from torch.nn import L1Loss
from torch.nn import CrossEntropyLoss

inputs = torch.tensor([1,2,3],dtype=torch.float32)
targets = torch.tensor([1,2,6],dtype=torch.float32)

inputs = torch.reshape(inputs,(1,1,1,3))
targets = torch.reshape(targets,(1,1,1,3))

loss = L1Loss(reduction='sum')
result = loss(inputs,targets)

print(result)

x = torch.tensor([0.1,0.2,0.3])
y = torch.tensor([1])
x = torch.reshape(x,(1,3))
loss_cross = CrossEntropyLoss()
loss = loss_cross(x,y)
print(loss)