import numpy as np
import sys


# write
def write_output(output_path,predict_column):
    with open(output_path,'w') as wf:
        for i in predict_column:
            wf.write(str(i)+'\n')   

def write_metric(metric_path,error_train,error_test):
    with open(metric_path,'w') as wf:
        wf.write(f'error(train): {error_train}'+'\n')
        wf.write(f'error(test): {error_test}'+'\n')

## read tsv
##load file
        
    data_array=np.loadtxt('xx.csv',delimiter=',')
    np.savetxt('xx.csv',data_array,delimiter=',') 

#append
        
    with open('para_data.csv', 'a') as file:
        np.savetxt(file, data_array)
        