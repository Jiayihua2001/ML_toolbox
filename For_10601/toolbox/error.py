import numpy as np
import sys

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
