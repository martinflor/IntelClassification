# -*- coding: utf-8 -*-
"""
Created on Tue May 10 22:28:57 2022

@author: Ineed
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

losses = [0.88, 0.58, 0.48, 0.43, 0.37]
train_acc = [0.66, 0.79, 0.83, 0.84, 0.87]
test_acc = [0.75, 0.78, 0.79, 0.82, 0.83]
x = [1,2,3,4,5]


fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Number of epochs')
ax1.set_ylabel('Loss')
ax1.set_xticks([1,2,3,4,5])
ax1.plot(x, losses, color=color)
ax1.tick_params(axis='y')

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
color2 = 'tab:orange'
ax2.set_ylabel("Training/Test accuracy")  # we already handled the x-label with ax1
ax2.plot(x, train_acc, color=color)
ax2.plot(x, test_acc, color = color2)
ax2.tick_params(axis='y')

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.grid()
#plt.title("Simple CNN on Intel Image Dataset")
plt.savefig("Loss-training.svg")
plt.show()