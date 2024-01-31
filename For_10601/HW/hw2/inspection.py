import numpy as np 
import sys

# Utilities

def load_file(input_path,format ='tsv'):
    """
    input_path: str 
        path of your input file
    format: str
        'csv' or 'tsv'
    return 
    """
    if format=='csv':
        sig=','
    if format=='tsv':
        sig='\t'
    with open(input_path,'r') as file:
        lines=file.readlines()
        attr_l=lines[0].strip().split(sig)[:-2]
        label=lines[0].strip().split(sig)[-1]
        attr={}
        data=[]
        for i in range(len(attr_l)):
            attr[attr_l[i]]=i
        for line in lines[1:]:
            cline = line.strip().split('\t')
            # line cleaned  ,for tsv - splited by tab ,return a list  
            row = [int(ele) for ele in cline]
            data.append(row) 
    return attr,np.array(data)


def count(D):
    unique_n, count = np.unique(D, return_counts=True)
    D_dic = dict(zip(unique_n, count))
    return D_dic

def H(D):
    """
    Entropy of array D.
    """
    D_dic = count(D)
    proportions = np.array(list(D_dic.values())) / len(D)
    E = -np.sum(proportions * np.log2(proportions))
    return E



def majority_voter(labels):
    """
    Determines the majority label in a list of labels. 
    If there is a tie, returns the label with the highest value.
    """
    unique_values, counts = np.unique(labels, return_counts=True)
    label_counts = dict(zip(unique_values, counts))
    max_count = max(label_counts.values())
    candidates = [label for label, count in label_counts.items() if count == max_count]
    major_ele = max(candidates)
    print(label_counts)
    return major_ele



def write_metric(metric_path,error_train,error_test):
    with open(metric_path,'w') as wf:
        wf.write(f'entropy: {error_train}'+'\n')
        wf.write(f'error: {error_test}')

# error calculator

def class_error(target_list,predict_column):
    t=0
    if len(target_list)==len(predict_column):
        for i in range(len(target_list)):
            if target_list[i] != predict_column[i]:
                t+=1
        error=t/len(target_list)
    else:
        print('the length of predicted data inconsistant with that of actual data')
    return error

            
            
#main route


if __name__ == '__main__':
    input_path_train = sys.argv[1]
    metric_path= sys.argv[2]
    
    print(f'input_path_train:{input_path_train}')
    print(f'metric_path:{metric_path}') 
    attr,data=load_file(input_path_train)
    labels=data[:,-1]
    major_ele = majority_voter(labels)
    predicted_labels=np.array([major_ele for i in range(len(labels))])
    entropy = H(labels)
    error_train = class_error(labels,predicted_labels)
    write_metric(metric_path,entropy,error_train)