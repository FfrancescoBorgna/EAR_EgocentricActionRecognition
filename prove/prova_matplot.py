import matplotlib.pyplot as plt
import numpy as np
import torch

vec = torch.tensor([1,1,2,3,1])
t = np.arange(0,5)

plt.figure()
plt.plot(t,vec)
plt.savefig("train_images/prova.png")
