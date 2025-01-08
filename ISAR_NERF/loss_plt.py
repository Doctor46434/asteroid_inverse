import matplotlib.pyplot as plt
import torch
import numpy as np

loss = torch.load("loss_list44.pth")
losses = [x.detach().cpu().numpy() for x in loss[500:]]
plt.plot(losses)
plt.xlabel('epoch')
plt.ylabel('loss function')
plt.show()
