import csv
import numpy as np
import argparse

VECTOR_LEN = 300   # Length of glove vector
MAX_WORD_LEN = 64  # Max word length in dict.txt and glove_embeddings.txt

################################################################################
# We have provided you the functions for loading the tsv and txt files. Feel   #
# free to use them! No need to change them at all.                             #
################################################################################


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
    dataset = np.loadtxt(file, delimiter='\t', comments=None, encoding='utf-8',
                         dtype='l,O')
    return dataset


def load_feature_dictionary(file):
    """
    Creates a map of words to vectors using the file that has the glove
    embeddings.

    Parameters:
        file (str): File path to the glove embedding file.

    Returns:
        A dictionary indexed by words, returning the corresponding glove
        embedding np.ndarray.
    """
    glove_map = dict()
    with open(file, encoding='utf-8') as f:
        read_file = csv.reader(f, delimiter='\t')
        for row in read_file:
            word, embedding = row[0], row[1:]
            glove_map[word] = np.array(embedding, dtype=float)
    return glove_map

def trim(str_i,glove_map):
    str_l=str_i.strip().split(' ')
    trim_list=[]
    for i in str_l:
        if i in glove_map.keys():
            trim_list.append(glove_map[i])
        else:
            pass
    return trim_list

def get_data(train_data,glove_map):
    N=len(train_data)
    data=[]
    for i in range(N):
        trim_list=trim(train_data[i][1],glove_map) 
        xi_feature=np.mean(np.array(trim_list),axis=0)
        Y_i=np.array(train_data[i][0])
        data_i=np.hstack((Y_i,xi_feature))
        data.append(data_i)
    return np.array(data)



if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the train_input path, you can use `args.train_input`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to training input .tsv file')
    parser.add_argument("validation_input", type=str, help='path to validation input .tsv file')
    parser.add_argument("test_input", type=str, help='path to the input .tsv file')
    parser.add_argument("feature_dictionary_in", type=str, 
                        help='path to the GloVe feature dictionary .txt file')
    parser.add_argument("train_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the training data should be written')
    parser.add_argument("validation_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the validation data should be written')
    parser.add_argument("test_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the test data should be written')
    args = parser.parse_args()


    # input data
    train_data_raw=load_tsv_dataset(args.train_input)
    test_data_raw=load_tsv_dataset(args.test_input)
    validation_data_raw=load_tsv_dataset(args.validation_input)
    glove_map =load_feature_dictionary(args.feature_dictionary_in)

    #extract features
    train_data=get_data(train_data_raw,glove_map)
    test_data=get_data(test_data_raw,glove_map)
    validation_data=get_data(validation_data_raw,glove_map)

    #write to file
    np.savetxt(args.train_out,train_data, delimiter='\t', fmt='%.6f')
    np.savetxt(args.test_out,test_data, delimiter='\t', fmt='%.6f')
    np.savetxt(args.validation_out,validation_data, delimiter='\t', fmt='%.6f')
