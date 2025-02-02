import matplotlib.pyplot as plt 
import numpy as np
import os

x= [i for i in range(0,9)]
y1= []
y2= []
def read_error():
    with open('heart_2_metrics.txt','r')as f:
        lines = f.readlines()
        line_train=lines[0].strip().split(':')[1].strip()
        line_test=lines[1].strip().split(':')[1].strip()
    return float(line_train),float(line_test)
   

for i in x:
    os.system(f'python decision_tree.py heart_train.tsv heart_test.tsv {i} \
heart_2_train.txt heart_2_test.txt heart_2_metrics.txt heart_2_print.txt')
    train_error,test_error =read_error()
    y1.append(train_error)
    y2.append(test_error)

fig, ax = plt.subplots()
ax.plot(x, y1, label='Train error')
ax.plot(x, y2, label='Test error')

ax.set(xlabel='max-depth', ylabel='error',
       title='heart.png')

ax.grid()
ax.legend()

plt.show()