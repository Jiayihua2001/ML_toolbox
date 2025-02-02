# HW3P2 Automatic Speech Recognition (ASR)
Jiayi Huang
email: jiayihua@andrew.cmu.edu
## Running code

Should change the datapath to the correct path in config.yaml first before run with this command:

`python hw4p2.py`

For dataset, make sure to unzip the data file downloaded from 
the autolab in the same content with hw3p2.py, data and file 
should be organized with the structure below:

````
handin:
    hw4p2.py
    utils.py
    config.yaml
    feature_representation
    masks.png
````

## Experiment and Hyperparameters

Here is the wandb link for hw4p2: 

https://wandb.ai/jiayihua-carnegie-mellon-university/HW4P2-Fall/workspace?nw=nwuserjiayihua

## Architecture of the final code

````

epochs: 60
batch_size: 32

mode                      : "full" # ["full", "dec_cond_lm", "dec_lm"]
pre_mode                  : "full" # ["full", "dec_cond_lm", "dec_lm"]
#dataset

token_type                : "char"     
feat_type                 : 'fbank'    
num_feats                 : 80 


# please check the default config.yaml for more details


