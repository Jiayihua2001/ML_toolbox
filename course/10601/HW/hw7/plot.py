import numpy as np
import matplotlib.pyplot as plt
import os

## step1 : first need to run this code to generate data  --generate data

# activation= "relu"
hidden_list=[64,128,512]
# for n_hi in hidden_list:
#     os.system(f'python rnn.py data/en.train_40.twocol.oov \
#     data/en.val_40.twocol.oov \
#     train_out.txt val_out.txt metrics_out.txt \
#     --activation="relu" --embedding_dim={n_hi} --hidden_dim={n_hi} \
#     --num_epochs=5')

## step2: read data and generate the plot

#load data you save from the step1



# for i in range(len(lr_l)):
#     lr=lr_l[i]
#     y_train = np.loadtxt(f'loss_lr{lr}_train',delimiter=',')
#     y_test = np.loadtxt(f'loss_lr{lr}_test',delimiter=',')
#     n_epoch=np.linspace(1,len(y_train),100)
#     fig,ax = plt.subplots()
#     ax.plot(n_epoch,y_train,label=f'Train_loss_{lr}')
#     ax.plot(n_epoch,y_test,label=f'Test_loss_{lr}')

#     ax.set(xlabel='number_of_epochs', ylabel='mean_cross_entropy_loss',
#         title=f'Fig_2_{lr}.png')
#     ax.grid()
#     ax.legend()
#     plt.show()

hidden_list=[64,128,512]
activation= "relu"
train_f1s=[]
test_f1s=[]
for hidden_dim in hidden_list:
    f1s=np.loadtxt(f'f1_{hidden_dim}_{activation}',delimiter=',')
    train_f1s.append(f1s[0,:])
    test_f1s.append(f1s[1,:])

n_epoch=np.linspace(1,5,5)
fig,ax = plt.subplots()
ax.plot(n_epoch,train_f1s[0],label=f'Train_f1_d64')
ax.plot(n_epoch,test_f1s[0],label=f'Test_f1_d64')
ax.plot(n_epoch,train_f1s[1],label=f'Train_f1_d128')
ax.plot(n_epoch,test_f1s[1],label=f'Test_f1_d128')
ax.plot(n_epoch,train_f1s[2],label=f'Train_f1_d512')
ax.plot(n_epoch,test_f1s[2],label=f'Test_f1_d512')
ax.set(xlabel='number_of_epochs', ylabel='F1_scores',
    title=f'Fig_2.png')
ax.grid()
ax.legend()
plt.show()