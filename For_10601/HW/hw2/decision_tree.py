import argparse
import numpy as np 

#neccessary utilities

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

##calc entropy 
def H(D):
    """
    Entropy of array D.
    """
    unique_n, count = np.unique(D, return_counts=True)
    D_dic = dict(zip(unique_n, count))
    proportions = np.array(list(D_dic.values())) / len(D)
    E = -np.sum(proportions * np.log2(proportions))
    return E

##split data
def test_split(data,attr,value=1):
    """
    get 2 splited datasets of labels according to attr A 

    Input:
    data: array
    attr: int 
        index of column of attr
    value: int 
        one value of attr which is the division criteria.

    return (array,array)
         list[0] is the array of labels which A<value.list[1]:else
    """
    left, right = list(), list()
    for row in data:
        if row[attr] < value:
            left.append(row)
        else:
            right.append(row)

    return np.array(left,ndmin=2), np.array(right,ndmin=2)

##count label of a dataset
def count_label(train_label,value=1):
    label_0=0
    label_1=1
    for i in train_label:
        if i < value:
            label_0+=1
        else: 
            label_1+=1
    return label_0,label_1

##calc the mutual information 
def I(data,attr,value):
    """
    Mutual information of label spliting on attr.
    """
    
    left_data,right_data=test_split(data,attr,value)
    if left_data.size ==0 or right_data.size ==0:
        I=0 
    else:
        tup_list=[(left_data.shape[0] / len(data[:,attr]), H(left_data[:,-1])),(right_data.shape[0] / len(data[:,attr]), H(right_data[:,-1]))]
        H_Y_A = np.array([i[0] * i[1] for i in tup_list])
        I = H(data[:,-1]) - np.sum(H_Y_A)
    return I

##majority voter
def majority_voter(label):
        """
        Determines the majority label in a list of labels. 
        If there is a tie, returns the label with the highest value.
        """
        unique_values, counts = np.unique(label, return_counts=True)
        label_counts = dict(zip(unique_values, counts))
        max_count = max(label_counts.values())
        candidates = [label for label, count in label_counts.items() if count == max_count]
        major_ele = max(candidates)
        return major_ele  




#Define node class(operation for leaf node)
class Node:
    def __init__(self, data=None, attr=None, subclass=None,depth=0,value=None):
        self.left = None
        self.right = None

        self.attr = attr
        self.labels = np.array(data,ndmin=2)[:,-1]  #label of data
        self.vote = None    # final label after majority vote if the node is leaf node
        self.data = data    # data :np.array  input dataset
        self.depth = depth  # depth of the current node ;for root node ,depth=0
        self.value = value  # division criteria for attr value
        self.subclass= subclass  # subclass=0,left node ,subclass =1,right node with respect to the parent attr

    def vote_leaf_node(self):
        if self.labels is not None:
            self.vote = majority_voter(self.labels)
        return self.vote
    def is_leaf(self):
        if self.left == None and self.right== None:
            return True
        else:
            return False

#build decision tree
        
def build_tree(data,attr_dic,max_depth,depth=0,subclass=None):
    #base case -pure -max_depth -attr run out
    if depth==max_depth or len(attr_dic)==0 or attr_dic is None:
        node=Node(data,depth=depth,subclass=subclass)
        node.vote_leaf_node()
        return node
    if data is None or type(data) is int:
        return 
    unique=np.unique(data[:,-1])
    if len(unique)==1 :
        node=Node(data,depth=depth,subclass=subclass)
        node.vote_leaf_node()
        return node
    
    #recuision:
    #current node
    # find the best split
    best_attr=None
    best_gain=-1000
    best_left=0
    best_right=0
    best_value=0
    for attr in attr_dic.keys():
        values=np.unique((data[:,attr])).tolist()
        if len(values)==1:
            pass
        else:
            for value in values:  
                if value > 0:
                    info_gain=I(data,attr,value)
                    if info_gain > best_gain:
                        best_gain=info_gain
                        best_attr=attr
                        best_value=value
                        best_left,best_right=test_split(data,attr,value)


    node=Node(data,attr=best_attr,depth=depth,value=best_value,subclass=subclass)

    left_attr = attr_dic.copy()
    right_attr = attr_dic.copy()
    if best_attr is not None:
        del left_attr[best_attr] 
        del right_attr[best_attr] 
    else: 
        node=Node(data,depth=depth,subclass=subclass)
        node.vote_leaf_node()
        return node

    node.left=build_tree(best_left,left_attr,max_depth=max_depth,depth=depth+1,subclass=0)
    node.right=build_tree(best_right,right_attr,max_depth=max_depth,depth=depth+1,subclass=1)           
    
    return node


#test decision tree 
def test_tree(tree,test_data):
    """
    for a single test point ,predict the result according to the decision tree trained
    """
    if tree.is_leaf():
        result=tree.vote
        return result
    elif tree.labels is None:
        return print('Blank Node')
    else:
        #current node -go down
        attr_value=test_data[tree.attr]
        print(attr_value)
        if attr_value < 1:
            return test_tree(tree.left,test_data)
        else:
            return test_tree(tree.right,test_data)
    

        
def collect_test(tree,test_dataset):
    """
    collect all the prediction of the test data point
    """
    test=[]
    for row in test_dataset:
        test.append(test_tree(tree,row))
    return np.array(test)


def print_tree(file,tree,attr_dic,last_attr=None):
    if tree is not None:
        if tree.depth==0:
            count_left,count_right=count_label(tree.labels)
            file.write(f'[{count_left}  0/{count_right} 1]'+'\n')
            if tree.attr==None:
                pass
            else:
                last_attr=attr_dic[tree.attr]
            #left
            print_tree(file,tree.left,attr_dic,last_attr)
            #right 
            print_tree(file,tree.right,attr_dic,last_attr)
        else:
            #current node ,non-root
            count_left,count_right=count_label(tree.labels)
            file.write('| '*(tree.depth)+f'{last_attr} = {tree.subclass}: [{count_left} 0/{count_right} 1]'+'\n')
            if tree.attr==None:
                pass
            else:
                last_attr=attr_dic[tree.attr]
            #left
            print_tree(file,tree.left,attr_dic,last_attr)
            #right 
            print_tree(file,tree.right,attr_dic,last_attr)



##error of classification problem
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
# write to file
            
def write_output(output_path,predict_column):
    with open(output_path,'w') as wf:
        for i in predict_column:
            wf.write(str(i)+'\n')   
            
def write_metric(metric_path,error_train,error_test):
    with open(metric_path,'w') as wf:
        wf.write(f'error(train): {error_train}'+'\n')
        wf.write(f'error(test): {error_test}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to training input .tsv file')
    parser.add_argument("test_input", type=str, help='path to the test input .tsv file')
    parser.add_argument("max_depth", type=int, help='maximum depth to which the tree should be built')
    parser.add_argument("train_out", type=str, help='path to output .txt file for training data predictions')
    parser.add_argument("test_out", type=str, help='path to output .txt file for test data predictions')
    parser.add_argument("metrics_out", type=str, help='path to output .txt file for train and test error metrics')
    parser.add_argument("print_out", type=str, help='path to print_out.txt file for printing decision trees')

    args = parser.parse_args()

    # Load Data
    attr_dic,train_data = load_file(args.train_input)
    test_data = load_file(args.test_input)[1]

    # Build Tree
    max_depth=args.max_depth
    tree = build_tree(train_data,attr_dic,max_depth)

    # Make Predictions
    predict_train = collect_test(tree,train_data)
    predict_test = collect_test(tree,test_data)

    # Calculate Errors
    train_error = class_error(train_data[:, -1], predict_train)
    test_error = class_error(test_data[:, -1], predict_test)

    # Write Outputs
    write_output(args.train_out, predict_train)
    write_output(args.test_out, predict_test)
    write_metric(args.metrics_out, train_error, test_error)
    with open(args.print_out,'w') as file:
        print_tree(file,tree,attr_dic)

if __name__ == '__main__':
  
    main()