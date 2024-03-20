import numpy as np
import matplotlib.pyplot as plt
import os

## step1 : first need to run this code to generate data  --generate data

lr_l=[0.03,0.003,0.0003]

for lr in lr_l:
    os.system(f'python neuralnet.py small_train.csv small_validation.csv small_train_out.labels small_validation_out.labels small_metrics_out.txt 100 50 1 {lr}')

## step2: read data and generate the plot

#load data you save from the step1

for i in range(len(lr_l)):
    lr=lr_l[i]
    y_train = np.loadtxt(f'loss_lr{lr}_train',delimiter=',')
    y_test = np.loadtxt(f'loss_lr{lr}_test',delimiter=',')
    n_epoch=np.linspace(1,len(y_train),100)
    fig,ax = plt.subplots()
    ax.plot(n_epoch,y_train,label=f'Train_loss_{lr}')
    ax.plot(n_epoch,y_test,label=f'Test_loss_{lr}')

    ax.set(xlabel='number_of_epochs', ylabel='mean_cross_entropy_loss',
        title=f'Fig_2_{lr}.png')
    ax.grid()
    ax.legend()
    plt.show()