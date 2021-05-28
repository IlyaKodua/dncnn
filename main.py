import Model

import torch
import torch.nn as nn
import numpy as np
import math
import matplotlib.pyplot as plt

model = Model.AutoEncoder()

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


epochs = 10000
loss_arr = np.zeros((epochs,1))
N = 50
for i in range(epochs):
    A = np.random.rand()
    w = np.random.rand() * math.pi/N
    f = np.random.rand() * 2 * math.pi
    b = (np.random.rand() - 0.5) * 1000
    # y =  torch.Tensor(A * np.sin( w * np.array(range(N)) + f) + b)
    y = torch.Tensor(np.array(range(N)))/(N-1)
    x = y + torch.Tensor(np.random.rand(N))/(N-1)

    x = torch.reshape(x, (1, 1, N))
    y = torch.reshape(y, (1, 1, N))
    y_hat = model.forward(x)
    loss = criterion(y_hat, y)
    loss_arr[i] = float(loss)
    if i % 10 == 0:
        print(f'Epoch: {i} Loss: {loss}')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
plt.figure(1)
plt.plot(loss_arr)
plt.show()


