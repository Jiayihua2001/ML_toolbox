import numpy as np
import argparse



def load_tsv_dataset(file):
    """
    Loads raw data and returns a tuple containing the reviews and their ratings.

    Parameters:
        file (str): File path to the dataset tsv file.

    Returns:
        An np.ndarray of shape N. N is the number of data points in the tsv file.
        Each element dataset[i] is a tuple (label, review), where the label is
        an integer (0 or 1) and the review is a string.
    """
    with open(file,'r') as file:
        lines=file.readlines()
        data=[]
        for line in lines:
            cline = line.strip().split('\t')
            # line cleaned  ,for tsv - splited by tab ,return a list  
            row = [float(ele) for ele in cline]
            data.append(row) 
            dataset=np.array(data)
    return dataset

def load_dataset(dataset):
    Y=dataset[:,0]
    data_copy=dataset.copy()
    data_copy[:,0]=1
    X=data_copy
    M=np.shape(X)[1]
    return X,Y

def sigmoid(theta,X):
        z=np.dot(X,theta)
        return 1/(1 + np.exp(-z))
   
def G_J(theta,X,Y):
    """
    gradient of negative likelihood 
    """
    fi=sigmoid(theta,X)
    G_J= np.dot(X,(fi-Y))
    return G_J


class logistic_regression():
    def __init__(self,dataset,num_epoch,learning_rate):
        X,Y=load_dataset(dataset)
        self.X=X
        self.Y=Y
        self.n_epoch=num_epoch
        self.learning_rate=learning_rate

    def train(self):
        """
        theta : np.ndarray, # shape (D,) where D is feature dim
        X : np.ndarray,     # shape (N, D) where N is num of examples
        y : np.ndarray,     # shape (N,)
        num_epoch : int, 
        learning_rate : float)"""
        theta = np.zeros(np.shape(self.X)[1])
        cur_epoch = 1
        X=self.X
        Y=self.Y
        N=np.shape(Y)[0]
        while (cur_epoch<=self.n_epoch):
            for i in range(N):
                Gradient_i=G_J(theta,X[i,:],Y[i])
                theta-= self.learning_rate*Gradient_i
            cur_epoch +=1
        self.theta_optimized=theta
        return 

    
    def predict(self,test_data):
        X,Y=load_dataset(test_data)
        N=len(Y)
        fi=sigmoid(self.theta_optimized,X)
    
        y_l=[]
        for i in range(N):
            if fi[i] >=0.5:
                y=1
                y_l.append(y)
            else:
                y=0
                y_l.append(y)

        y_pred=np.array(y_l).astype(np.int64)
    
        error = compute_error(y_pred,Y)

        return error,y_pred
    

def compute_error(y_pred,Y):
    error=0
    N=len(Y)
    for i in range(N):
        if y_pred[i] != Y[i]:
            error+=1
        else:
            pass
    return float(error/N)

        
def write_output(output_path,y_pred):
    np.savetxt(output_path, y_pred,fmt='%s')
            
def write_metric(metric_path,error_train,error_test):
    with open(metric_path,'w') as wf:
        wf.write(f'error(train): {error_train}'+'\n')
        wf.write(f'error(test): {error_test}')




if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the learning rate, you can use `args.learning_rate`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to formatted training data')
    parser.add_argument("validation_input", type=str, help='path to formatted validation data')
    parser.add_argument("test_input", type=str, help='path to formatted test data')
    parser.add_argument("train_out", type=str, help='file to write train predictions to')
    parser.add_argument("test_out", type=str, help='file to write test predictions to')
    parser.add_argument("metrics_out", type=str, help='file to write metrics to')
    parser.add_argument("num_epoch", type=int, 
                        help='number of epochs of stochastic gradient descent to run')
    parser.add_argument("learning_rate", type=float,
                        help='learning rate for stochastic gradient descent')
    args = parser.parse_args()

    train_data=load_tsv_dataset(args.train_input)
    val_data=load_tsv_dataset(args.validation_input)
    test_data=load_tsv_dataset(args.test_input)

    n_epoch = args.num_epoch
    l_r=args.learning_rate



    lr=logistic_regression(train_data,n_epoch,l_r)
    lr.train()
    train_error,train_y_pred=lr.predict(train_data)
    val_error,val_y_pred=lr.predict(val_data)
    test_error,test_y_pred=lr.predict(test_data)


    write_output(args.train_out,train_y_pred)
    write_output(args.test_out,test_y_pred)
    write_metric(args.metrics_out,train_error,test_error)


    
        





