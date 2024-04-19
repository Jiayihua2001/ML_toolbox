import numpy as np
import matplotlib.pyplot as plt
import os

## step1 : first need to run this code to generate data  --generate data

# lr_l=[0.03,0.003,0.0003]

# for lr in lr_l:
#     os.system(f'python neuralnet.py small_train.csv small_validation.csv small_train_out.labels small_validation_out.labels small_metrics_out.txt 100 50 1 {lr}')

## step2: read data and generate the plot

#load data you save from the step1

# epoch_num = 2500
# n_epoch = np.linspace(0,epoch_num ,epoch_num)
# y = np.loadtxt('mc_raw_returns.txt',delimiter=' ')
# window = 25
# n_window = np.linspace(0,epoch_num-window,epoch_num-window)

# y_roll=[]
# for i in range(len(n_window)):
#     roll_all = 0
#     for n in range(window):
#         roll_all += y[i+n]
#     roll_mean = roll_all/window
#     y_roll.append(roll_mean)

# fig,ax = plt.subplots()
# ax.plot(n_epoch,y,label='returns of raw model')

# ax.plot(n_window,y_roll,label='rolling mean of raw model')

# ax.set(xlabel='number_of_epochs', ylabel='returns per epoch',
#     title=f'Fig_raw.png')
# ax.grid()
# ax.legend()
# plt.show()





epoch_num = 400
n_epoch = np.linspace(0,epoch_num ,epoch_num)
y = np.loadtxt('mc_tile_returns.txt',delimiter=' ')
window = 25
n_window = np.linspace(0,epoch_num-window,epoch_num-window)

y_roll=[]
for i in range(len(n_window)):
    roll_all = 0
    for n in range(window):
        roll_all += y[i+n]
    roll_mean = roll_all/window
    y_roll.append(roll_mean)

fig,ax = plt.subplots()
ax.plot(n_epoch,y,label='returns of tile model')

ax.plot(n_window,y_roll,label='rolling mean of tile model')

ax.set(xlabel='number_of_epochs', ylabel='returns per epoch',
    title=f'Fig_tile.png')
ax.grid()
ax.legend()
plt.show()