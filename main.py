import Model

import torch
import torch.nn as nn
import numpy as np
import math
import matplotlib.pyplot as plt

model = Model.AutoEncoder()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


epochs = 10000
loss_arr = np.zeros((epochs,1))
N = 50
for i in range(epochs):
    A = np.random.rand()
    w = np.random.rand() * math.pi/N
    f = np.random.rand() * 2 * math.pi
    b = (np.random.rand() - 0.5) * 1000
    # y =  torch.Tensor(A * np.sin( w * np.array(range(N)) + f) + b)
    y = A*np.array(range(N)) + b + f*np.array(range(N))*np.array(range(N))
    y  -= np.mean(y)
    y = torch.Tensor(y)
    x = y + A*0.2*torch.Tensor(A*np.random.rand(N))

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

