import Model

import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

dir_path = 'dataset/'

<<<<<<< HEAD
transform = transforms.Compose ([transforms.ToTensor (),
                                 transforms.Grayscale(),
                                 transforms.Normalize((127.5, 127.5, 127.5),
                                                      (127.5, 127.5, 127.5)),
                                 ])
=======
transform = transforms.Compose ([transforms.Grayscale(num_output_channels=1),
# transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
transforms.ToTensor ()])
>>>>>>> f7a14676b961b4fcfa2c28604d5d5da1183ce223


dataset = datasets.ImageFolder(root=dir_path, transform=transform)

<<<<<<< HEAD
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

model = Model.DNCNN(1, 64, 3)
criterion = nn.MSELoss()
=======
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)

model = Model.DNCNN(1, 64, 3)
model.to(device)
criterion = nn.MSELoss(reduction='mean')
>>>>>>> f7a14676b961b4fcfa2c28604d5d5da1183ce223
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = len(dataloader)
loss_arr = np.zeros((epochs,1))
cnt = 0
for i in range(epochs):
    dataiter = iter(dataloader)
    im, labels = dataiter.next()
    # im = images[0]
    optimizer.zero_grad()
    noise = torch.zeros(im.size())
    stdN = np.random.uniform(0, 55, size=noise.size()[0])
    for n in range(noise.size()[0]):
        sizeN = noise[0,:,:,:].size()
        noise[n,:,:,:] = torch.FloatTensor(sizeN).normal_(mean=0, std=stdN[n]/255.)
    im_in = im + noise
    out = model(im_in.to(device))
    loss = criterion(out, noise.to(device))
    loss_arr[cnt] = loss.item()
    loss.backward()
    optimizer.step()
    print(cnt)
    print(loss_arr[cnt])
    cnt += 1

PATH = 'net.pth'
torch.save(model.state_dict(), PATH)

plt.figure(2)
plt.plot(loss_arr, 'r')
plt.show()

