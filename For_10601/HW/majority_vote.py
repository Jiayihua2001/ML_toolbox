import numpy as np
import sys

def get_target_column(input_path,target_column):
    """
    input_path: str
    target_column: str 
        name of your target column
    return list of the last column of tsv file
    """
    target_list=[]
    
    with open(input_path,'r') as file:
        for line in file.readlines():
            cline = line.strip().split('\t') # line cleaned  ,for tsv - splited by tab ,return a list
            if target_column in cline :
                print(f'target column: {target_column}')
                for i in range(len(cline)):
                    if cline[i]== target_column :
                        target_index=i
            else:
                target_list.append(int(cline[i]))
        print(f'length of target column: {len(target_list)}' )
    return target_list

def majority_voter(target_list):
    count_dic={}
    predict_column=[]
    #unique value -initialization
    unique_value_l = list(set(target_list))
    for ele in unique_value_l:  
        count_dic[ele] = 0
    # find and count value in the target list
    for ele in target_list:
        count_dic[ele] +=1
    max_count = max(list(count_dic.values()))
    print(f'count_dic: {count_dic}')
    major_ele = [key for key,value in count_dic.items() if value == max_count][0]
    print(f'major_element: {major_ele}')
    for i in range(len(target_list)):
        predict_column.append(major_ele)
    return predict_column

#write files

def write_output(output_path,predict_column):
    with open(output_path,'w') as wf:
        for i in predict_column:
            wf.write(str(i)+'\n')    

def write_metric(metric_path,error_train,error_test):
    with open(metric_path,'w') as wf:
        wf.write(f'error(train): {error_train}'+'\n')
        wf.write(f'error(test): {error_test}'+'\n')

# error calculator

def MSE_error(target_list,predict_column):
    """
    calculate the Mean Squared Error (MSE)
    """
    t=0
    if len(target_list)==len(predict_column):
        for i in range(len(target_list)):
            t+=(target_list[i]-predict_column[i])**2
        MSE =t/len(target_list)
    else:
        print('the length of predicted data inconsistant with that of actual data')
    return MSE
def MAE_error(target_list,predict_column):
    """
    calculate the Mean Absolute Error (MAE)
    """
    t=0
    if len(target_list)==len(predict_column):
        for i in range(len(target_list)):
            t+=abs(target_list[i]-predict_column[i])
        MAE =t/len(target_list)
    else:
        print('the length of predicted data inconsistant with that of actual data')
    return MAE
def RMSE_error(target_list,predict_column):
    """
    calculate the Root Mean Squared Error (RMSE)
    """
    t=0
    if len(target_list)==len(predict_column):
        for i in range(len(target_list)):
            t+=(target_list[i]-predict_column[i])**2
        RMSE =np.sqrt(t/len(target_list))
    else:
        print('the length of predicted data inconsistant with that of actual data')
    return RMSE

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

    target_column = 'grade'

    target_list_train = get_target_column(input_path_train,target_column)
    predict_column_train = majority_voter(target_list_train)
    error_train = class_error(target_list_train,predict_column_train)
    write_output(output_path_train, predict_column_train)

    target_list_test = get_target_column(input_path_test,target_column)
    predict_column_test = majority_voter(target_list_test)
    write_output(output_path_test, predict_column_test)
    error_test = class_error(target_list_test,predict_column_test)

    write_metric(metric_path,error_train,error_test)
   

