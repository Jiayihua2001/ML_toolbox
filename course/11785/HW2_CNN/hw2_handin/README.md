# HW2P2 Face Recognition & Face Verification

Jiayi Huang
Andrew ID: jiayihua

## Running code

load kernel environment  with pytorch
Run notebook : hw2-densenet.ipynb

For dataset, make sure to unzip the data file downloaded from 
the autolab in the same content with hw2p2.py, data and file 
should be organized with the structure below:

````
handin:
    hw2-densenet.ipynb
    data:
        11-785-f22-hw2p2-classification:
            classification:
                dev
                test
                train
                train_subset
                classification_sample_submission.csv
        verification:
            known
            unknown_dev
            unknown_test
            dev_identities.csv            
````

## Experiment and Hyperparameters

Here is the wandb link for hw2p2:  https://wandb.ai/jiayihua-carnegie-mellon-university/hw2p2-ablations/workspace?nw=nwuserjiayihua

## Architecture of the final code

````
batch_size:512
bottleneck:true
checkpoint_dir:"checkpoint"
data_dir:"/global/cfs/cdirs/m3578/jiayihua/11785/data/11-785-f24-hw2p2-verification/cls_data"
data_ver_dir:"/global/cfs/cdirs/m3578/jiayihua/11785/data/11-785-f24-hw2p2-verification/ver_data"
drop_rate:0.2
epochs:160
growth_rate:32
label_smoothing:0.1
lr:0.0001
model:"Densenet-3"
    0:6
    1:12
    2:24
    3:48
num_blocks:4
num_classes:8,631
````
## summary metrics
````
train_cls_acc:98.84536441789405
train_loss:1.4844992149484255
valid_cls_acc:90.94034486658433
valid_loss:2.031521907974692
valid_ret_acc:86.4
valid_ret_err:13.7
````