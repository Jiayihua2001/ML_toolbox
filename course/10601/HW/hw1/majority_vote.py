import numpy as np
import sys

def get_target_column(input_path):
    """
    input_path: str
    target_column: str 
        name of your target column
    return list of the last column of tsv file
    """
    target_list=[]
    with open(input_path,'r') as file:
        lines=file.readlines()
        target_column=lines[0].strip().split('\t')[-1]
        for line in lines[1:]:
            cline = line.strip().split('\t') # line cleaned  ,for tsv - splited by tab ,return a list
            target_list.append(int(cline[-1]))
        print(f'length of {target_column}: {len(target_list)}' )
    return target_list

def majority_voter(target_list):
    """
    target_list:list
    equal_ele : when the elements all have the same amount,which element should return
    """
    count_dic={}
    predict_column=[]
    #unique value -initialization
    unique_value_l = list(set(target_list))
    for ele in unique_value_l:  
        count_dic[ele] = 0
    # find and count value in the target list
    for ele in target_list:
        count_dic[ele] +=1
    if max(list(count_dic.values())) == min(list(count_dic.values())):
        major_ele = max(unique_value_l)
        print(f'major_element: {major_ele}')
    else:
        max_count = max(list(count_dic.values()))
        print(f'count_dic: {count_dic}')
        major_ele = [key for key,value in count_dic.items() if value == max_count][0]
        print(f'major_element: {major_ele}')
    for i in range(len(target_list)):
        predict_column.append(major_ele)
    return predict_column

def train_model(target_list_test,predict_value):
    predict_column=[]
    for i in target_list_test:
        predict_column.append(predict_value)
    return predict_column
        
#write files

def write_output(output_path,predict_column):
    with open(output_path,'w') as wf:
        for i in predict_column:
            wf.write(str(i)+'\n')    

def write_metric(metric_path,error_train,error_test):
    with open(metric_path,'w') as wf:
        wf.write(f'error(train): {error_train}'+'\n')
        wf.write(f'error(test): {error_test}')

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

            
            



if __name__ == '__main__':
    input_path_train = sys.argv[1]
    input_path_test = sys.argv[2]
    output_path_train = sys.argv[3]
    output_path_test = sys.argv[4]
    metric_path = sys.argv[5]
    
    print(f'input_path_train:{input_path_train}')
    print(f'input_path_test:{input_path_test}')
    print(f'output_path_train:{output_path_train}')
    print(f'output_path_test:{output_path_test}')
    print(f'metric_path:{metric_path}') 
    # main route


    print('Training set info:')
    target_list_train = get_target_column(input_path_train)
    predict_column_train = majority_voter(target_list_train)
    error_train = class_error(target_list_train,predict_column_train)
    write_output(output_path_train, predict_column_train)
    print('Testing set info:')
    target_list_test = get_target_column(input_path_test)
    predict_column_test = train_model(target_list_test,predict_column_train[0])
    write_output(output_path_test, predict_column_test)
    error_test = class_error(target_list_test,predict_column_test)

    write_metric(metric_path,error_train,error_test)
   

