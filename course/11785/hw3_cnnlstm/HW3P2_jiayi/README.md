# HW3P2 Automatic Speech Recognition (ASR)
Jiayi Huang
email: jiayihua@andrew.cmu.edu
## Running code

Should change the datapath to the correct path in config.yaml first before run with this command:

`python hw3p2.py`

For dataset, make sure to unzip the data file downloaded from 
the autolab in the same content with hw3p2.py, data and file 
should be organized with the structure below:

````
handin:
    hw3p2.py
    CNNLSTM.py
    config.yaml
    ml_utils.py 
````

## Experiment and Hyperparameters

Here is the wandb link for hw3p2: 

https://wandb.ai/jiayihua-carnegie-mellon-university/hw3_p2_ablation?nw=nwuserjiayihua

## Architecture of the final code

````
epochs: 100
batch_size: 128
learning_rate: 0.002
lstm dropout: 0.4
hidden dim: 512
beam width: 2

Model architecture: CNN and LSTM
channels: [32, 64, 128, 256]

loss: The Connectionist Temporal Classification (CTC)loss
activation: GELU, SiLU
Batchnorm: BatchNorm1d
Optimizer:Adam
Learning rate schedule: ReduceLROnPlateau
````





