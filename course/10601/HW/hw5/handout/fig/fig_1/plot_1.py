import numpy as np
import matplotlib.pyplot as plt
import os

#Fig_1

x = [5,20,50,100,200]


# for hu in x:
#     os.system(f'python neuralnet.py small_train.csv small_validation.csv small_train_out.labels small_validation_out.labels small_metrics_out.txt 100 {hu} 1 0.001')

data=np.loadtxt('n_hid.txt',delimiter=',')
print(data)
y_train=data[:,1]
y_test=data[:,2]

fig,ax = plt.subplots()
ax.plot(x,y_train,label='Train_loss')
ax.plot(x,y_test,label='Test_loss')

ax.set(xlabel='number_of_hidden_units', ylabel='cross_entropy_loss',
       title='Fig1.png')
ax.grid()
ax.legend()

plt.show()


