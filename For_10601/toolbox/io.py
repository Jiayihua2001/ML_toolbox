import numpy as np
import sys

# read tsv
##load file

def load_file(input_path,format ='tsv'):
    """
    input_path: str 
        path of your input file
    format: str
        'csv' or 'tsv'
    return attr_dic :key=index of attr ,value=attr
            dataset:numpy array of the whole dataset.
    """
    if format=='csv':
        sig=','
    if format=='tsv':
        sig='\t'
    with open(input_path,'r') as file:
        lines=file.readlines()
        attr_l=lines[0].strip().split(sig)[:-1]
        attr_dic={}
        data=[]
        for i,attr in enumerate(attr_l):
            attr_dic[i]=attr
        for line in lines[1:]:
            cline = line.strip().split('\t')
            # line cleaned  ,for tsv - splited by tab ,return a list  
            row = [int(ele) for ele in cline]
            data.append(row) 
            dataset=np.array(data)
    return attr_dic,dataset

def get_target_column_csv(input_path,target_column,):
    """
    input_path: str . path of csv file
    target_column: str 
        name of your target column
    return list of the target column of csv file
    """
    target_list=[]
    
    with open(input_path,'r') as file:
        for line in file.readlines():
            cline = line.strip().split(',') # line cleaned  ,for tsv - splited by tab ,return a list
            if target_column in cline :
                print(f'target column: {target_column}')
                for i in range(len(cline)):
                    if cline[i]== target_column :
                        target_index=i
            else:
                target_list.append(int(cline[i]))
        print(f'length of target column: {len(target_list)}' )
    return target_list


# write
def write_output(output_path,predict_column):
    with open(output_path,'w') as wf:
        for i in predict_column:
            wf.write(str(i)+'\n')    

def write_metric(metric_path,error_train,error_test):
    with open(metric_path,'w') as wf:
        wf.write(f'error(train): {error_train}'+'\n')
        wf.write(f'error(test): {error_test}'+'\n')