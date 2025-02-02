
import torch
import torch.nn as nn
# import torch.nn.functional as F
import torchaudio.functional as aF
import torchaudio.transforms as tat
# from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import TensorDataset, Dataset, DataLoader

import gc
import os
from transformers import AutoTokenizer
import yaml
import math
from typing import Literal, List, Optional, Any, Dict, Tuple
import random
# import zipfile
# import datetime
from torchinfo import summary
# import glob
import wandb
import numpy as np
# import pandas as pd
from tqdm import tqdm
from scipy.fftpack import dct
import seaborn as sns
import matplotlib.pyplot as plt
''' Imports for decoding and distance calculation. '''
import json
import warnings
import shutil
warnings.filterwarnings("ignore")


from utils import *

#device

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

print("Device: ", device)

#set random seed
random.seed(11)
np.random.seed(11)
torch.manual_seed(11)
torch.cuda.manual_seed(11)

# load config
with open("config.yaml") as file:
    config = yaml.safe_load(file)


#initialize tokenizer
Tokenizer = GTokenizer(config['token_type'])



# load data
## load dataset
train_dataset = SpeechDataset(
    partition = config['train_partition'],
    config = config,
    tokenizer = Tokenizer,
    isTrainPartition = True,
)

if config["mode"] in ("full", "dec_cond_lm"):
    val_dataset     = SpeechDataset(
        partition   = config['val_partition'],
        config      = config,
        tokenizer   = Tokenizer,
        isTrainPartition = False,
    )

if config["mode"] == "full":
    test_dataset    = SpeechDataset(
        partition   = config['test_partition'],
        config      = config,
        tokenizer   = Tokenizer,
        isTrainPartition = False,
    )

text_dataset   = TextDataset(
     partition  = config['unpaired_text_partition'],
     config     = config,
     tokenizer  = Tokenizer,
)


## DataLoaders
train_loader    = DataLoader(
    dataset     = train_dataset,
    batch_size  = config["batch_size"],
    shuffle     = True,
    num_workers = config['NUM_WORKERS'],
    pin_memory  = True,
    collate_fn  = train_dataset.collate_fn
)

val_loader = None
if config["mode"] in ("full", "dec_cond_lm"):
    val_loader      = DataLoader(
        dataset     = val_dataset,
        batch_size  = 4,
        shuffle     = False,
        num_workers = config['NUM_WORKERS'],
        pin_memory  = True,
        collate_fn  = val_dataset.collate_fn
    )

test_loader = None
if config["mode"] == "full":
    test_loader     = DataLoader(
        dataset     = test_dataset,
        batch_size  = config["batch_size"],
        shuffle     = False,
        num_workers = config['NUM_WORKERS'],
        pin_memory  = True,
        collate_fn  = test_dataset.collate_fn
    )

text_loader = None
# UNCOMMENT if pretraining decoder as LM
# if config["mode"] == "dec_lm":
#     text_loader     = DataLoader(
#         dataset       = text_dataset,
#         batch_size    = config["batch_size"],
#         shuffle       = True,
#         num_workers   = config['NUM_WORKERS'],
#         pin_memory    = True,
#         collate_fn    = text_dataset.collate_fn
#     )

def verify_dataset(dataloader, partition):
    '''Compute the Maximum MFCC and Transcript sequence length in a dataset'''
    print("Loaded Path: ", partition)
    max_len_feat = 0
    max_len_t    = 0  # To track the maximum length of transcripts

    # Iterate through the dataloader
    for batch in tqdm(dataloader, desc=f"Verifying {partition} Dataset"):
      try:
        x_pad, y_shifted_pad, y_golden_pad, x_len, y_len = batch

        # Update the maximum feat length
        len_x = x_pad.shape[1]
        if len_x > max_len_feat:
            max_len_feat = len_x

        # Update the maximum transcript length
        # transcript length is dim 1 of y_shifted_pad
        if y_shifted_pad is not None:
          len_y = y_shifted_pad.shape[1]
          if len_y > max_len_t:
              max_len_t = len_y

      except Exception as e:
        # The text dataset has no transcripts
        y_shifted_pad, y_golden_pad, y_len = batch

        # Update the maximum transcript length
        # transcript length is dim 1 of y_shifted_pad
        len_y = y_shifted_pad.shape[1]
        if len_y > max_len_t:
            max_len_t = len_y


    print(f"Maximum Feat Length in Dataset       : {max_len_feat}")
    print(f"Maximum Transcript Length in Dataset : {max_len_t}")
    return max_len_feat, max_len_t

print('')
print("Paired Data Stats: ")
print(f"No. of Train Feats   : {train_dataset.__len__()}")
print(f"Batch Size           : {config['batch_size']}")
print(f"Train Batches        : {train_loader.__len__()}")
print(f"Val Batches          : {val_loader.__len__()}")
print(f"Test Batches         : {test_loader.__len__()}")
print('')
print("Checking the Shapes of the Data --\n")
for batch in train_loader:
    x_pad, y_shifted_pad, y_golden_pad, x_len, y_len, = batch
    print(f"x_pad shape:\t\t{x_pad.shape}")
    print(f"x_len shape:\t\t{x_len.shape}")

    if y_shifted_pad is not None and y_golden_pad is not None and y_len is not None:
      print(f"y_shifted_pad shape:\t{y_shifted_pad.shape}")
      print(f"y_golden_pad shape:\t{y_golden_pad.shape}")
      print(f"y_len shape:\t\t{y_len.shape}\n")
      # convert one transcript to text
      transcript = train_dataset.tokenizer.decode(y_shifted_pad[0].tolist())
      print(f"Transcript Shifted: {transcript}")
      transcript = train_dataset.tokenizer.decode(y_golden_pad[0].tolist())
      print(f"Transcript Golden: {transcript}")
    break
print('')
'''
# UNCOMMENT if pretraining decoder as LM
print("Unpaired Data Stats: ")
print(f"No. of text          : {text_dataset.__len__()}")
print(f"Batch Size           : {config['batch_size']}")
print(f"Train Batches        : {text_loader.__len__()}")
print('')
print("Checking the Shapes of the Data --\n")
for batch in text_loader:
     y_shifted_pad, y_golden_pad, y_len, = batch
     print(f"y_shifted_pad shape:\t{y_shifted_pad.shape}")
     print(f"y_golden_pad shape:\t{y_golden_pad.shape}")
     print(f"y_len shape:\t\t{y_len.shape}\n")

     # convert one transcript to text
     transcript = text_dataset.tokenizer.decode(y_shifted_pad[0].tolist())
     print(f"Transcript Shifted: {transcript}")
     transcript = text_dataset.tokenizer.decode(y_golden_pad[0].tolist())
     print(f"Transcript Golden: {transcript}")
     break
print('')
'''
print("\n\nVerifying Datasets")
max_train_feat, max_train_transcript = verify_dataset(train_loader, config['train_partition'])
max_val_feat, max_val_transcript     = verify_dataset(val_loader,   config['val_partition'])
max_test_feat, max_test_transcript   = verify_dataset(test_loader,  config['test_partition'])
#_, max_text_transcript               = verify_dataset(text_loader,  config['unpaired_text_partition'])

MAX_SPEECH_LEN = max(max_train_feat, max_val_feat, max_test_feat)
MAX_TRANS_LEN  = max(max_train_transcript, max_val_transcript)
print(f"Maximum Feat. Length in Entire Dataset      : {MAX_SPEECH_LEN}")
print(f"Maximum Transcript Length in Entire Dataset : {MAX_TRANS_LEN}")
print('')

gc.collect()

plt.figure(figsize=(10, 6))
plt.imshow(x_pad[0].numpy().T, aspect='auto', origin='lower', cmap='viridis')
plt.xlabel('Time')
plt.ylabel('Features')
plt.title('Feature Representation')
plt.show()


#MASK TEST
# Test w/ dummy inputs

enc_inp_tensor     = torch.randn(4, 20, 32)  # (N, T,  *)
dec_inp_tensor     = torch.randn(4, 10)       # (N, T', *)
enc_inp_lengths    = torch.tensor([13, 6, 9, 12])   # Lengths of input sequences before padding
dec_inp_lengths    = torch.tensor([8, 3, 1, 6])   # Lengths of target sequences before padding


enc_padding_mask        = PadMask(padded_input=enc_inp_tensor, input_lengths=enc_inp_lengths)
dec_padding_mask        = PadMask(padded_input=dec_inp_tensor, input_lengths=dec_inp_lengths)
dec_causal_mask         = CausalMask(input_tensor=dec_inp_tensor)

# Black portions are attended to
fig, axs = plt.subplots(1, 3, figsize=(20, 5))
axs[0].imshow(enc_padding_mask, cmap="gray", aspect='auto')
axs[0].set_title("Encoder Padding Mask")
axs[1].imshow(dec_padding_mask, cmap="gray", aspect='auto')
axs[1].set_title("Decoder Padding Mask")
axs[2].imshow(dec_causal_mask, cmap="gray", aspect='auto')
axs[2].set_title("Decoder Causal Self-Attn Mask")

plt.savefig("masks.png")



# model

model = Transformer(
    target_vocab_size = Tokenizer.VOCAB_SIZE,
    d_model        = config['d_model'],
    d_ff           = config['d_ff'],
    initialization = config['initialization'],
    std            = config['std'],

    input_dim      = x_pad.shape[-1],
    time_stride    = config['time_stride'],
    feature_stride = config['feature_stride'],
    embed_dropout  = config['embed_dropout'],

    enc_num_layers = config['enc_num_layers'],
    enc_num_heads  = config['enc_num_heads'],
    speech_max_len = MAX_SPEECH_LEN,
    enc_dropout    = config['enc_dropout'],

    dec_num_layers = config['dec_num_layers'],
    dec_num_heads  = config['dec_num_heads'],
    dec_dropout    = config['dec_dropout'],
    trans_max_len  = MAX_TRANS_LEN
)


model.to(device)
x_pad = x_pad.to(device)
x_len = x_len.to(device)
y_shifted_pad = y_shifted_pad.to(device)
y_len = y_len.to(device)

summary(model, input_data=[x_pad, x_len, y_shifted_pad, y_len])
gc.collect()
torch.cuda.empty_cache()

# Loss,Optim,Scheduler
loss_func   = nn.CrossEntropyLoss(ignore_index = Tokenizer.PAD_TOKEN)
ctc_loss_fn  = None
if config['use_ctc']:
    ctc_loss_fn = nn.CTCLoss(blank=Tokenizer.PAD_TOKEN)
scaler = torch.cuda.amp.GradScaler()

optimizer = get_optimizer(config, model)
assert optimizer!=None

scheduler = get_scheduler(config, optimizer)

#load wandb

# using WandB? resume training?
USE_WANDB = config['use_wandb']
RESUME_LOGGING = False

# creating your WandB run
run_name = "{}_{}_Transformer_ENC-{}-{}_DEC-{}-{}_{}_{}_{}_{}_token_{}".format(
    config["Name"],
    config['feat_type'],
    config["enc_num_layers"],
    config["enc_num_heads"],
    config["dec_num_layers"],
    config["dec_num_heads"],
    config["d_model"],
    config["d_ff"],
    config["optimizer"],
    config["scheduler"],
    config["token_type"],
    )

expt_root = os.path.join(os.getcwd(), run_name)
os.makedirs(expt_root, exist_ok=True)

if USE_WANDB:
    wandb.login(key="bc022b99e5a39b97fc6ae8c641ab328e9f52d2e6", relogin=True) # TODO enter your key here
    run = wandb.init(
        name    = config["Name"],     ### Wandb creates random run names if you skip this field, we recommend you give useful names
        reinit  = True,         ### Allows reinitalizing runs when you re-run this cell
        project = "HW4P2-Fall",  ### Project should be created in your wandb account
        config  = config        ### Wandb Config for your run
    )

### Save your model architecture as a string with str(model)
model_arch  = str(model)
### Save it in a txt file
model_path = os.path.join(expt_root, "model_arch.txt")
arch_file   = open(model_path, "w")
file_write  = arch_file.write(model_arch)
arch_file.close()

### Log it in your wandb run with wandb.sav
### Create a local directory with all the checkpoints
idx = 1
while os.path.exists(os.path.join(expt_root, f"config_{idx}.yaml")):
    idx += 1
shutil.copy(os.path.join(os.getcwd(), 'config.yaml'), os.path.join(expt_root, f"config_{idx}.yaml"))
e                   = 0
best_loss           = float('inf')
best_perplexity     = float('inf')
best_cer            = float('inf')
RESUME_LOGGING  = config["resume"]
checkpoint_root = os.path.join(expt_root, 'checkpoints')
text_root       = os.path.join(expt_root, 'out_text')
attn_img_root   = os.path.join(expt_root, 'attention_imgs')
os.makedirs(checkpoint_root, exist_ok=True)
os.makedirs(attn_img_root,   exist_ok=True)
os.makedirs(text_root,       exist_ok=True)
checkpoint_best_loss_model_filename     = f'checkpoint-best-{config["mode"]}.pth'
checkpoint_last_epoch_filename          = f'checkpoint-{config["mode"]}-'
best_loss_model_path                    = os.path.join(checkpoint_root, checkpoint_best_loss_model_filename)


if USE_WANDB:
    wandb.watch(model, log="all")

if RESUME_LOGGING:
    # change if you want to load best test model accordingly
    # checkpoint = torch.load(wandb.restore(checkpoint_best_loss_model_filename, run_path=""+run_id).name)
    checkpoint = torch.load(best_loss_model_path, map_location = device)
    model.load_state_dict(checkpoint['model_state_dict'], strict = False)
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    e = checkpoint['epoch'] + 1
    assert 'perplexity' in checkpoint, "Perplexity not found in checkpoint"
    best_perplexity = checkpoint['perplexity']
    if 'CER' in checkpoint:
        best_cer = checkpoint['CER']

    print("Resuming from epoch {}".format(e+1))
    print("Total epochs as pretrain: ", config['pretrain_epochs'])
    print("Total epochs as train: ", config["train_epochs"])
    print("Optimizer: \n", optimizer)
    print("Best Perplexity: ", best_perplexity)
    print("Best CER: ", best_cer)

torch.cuda.empty_cache()
gc.collect()


# pretrain approach 1: decoder lm

if config["mode"] == "dec_lm":


    epochs = config["pretrain_epochs"]

    for epoch in range(e, epochs):

        print("\nEpoch {}/{}".format(epoch+1, epochs))

        curr_lr = float(optimizer.param_groups[0]["lr"])

        train_loss, train_perplexity, attention_weights = train_step(
            model,
            criterion=loss_func,
            ctc_loss=None,
            ctc_weight=0.,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            train_loader=text_loader,
            tokenizer=Tokenizer,
            mode='dec_lm',
            config=config
        )

        print("\nEpoch {}/{}: \nTrain Loss {:.04f}\t Train Perplexity {:.04f}\t Learning Rate {:.06f}".format(
            epoch + 1, epochs, train_loss, train_perplexity, curr_lr))

        attention_keys = list(attention_weights.keys())
        attention_weights_decoder_self       = attention_weights[attention_keys[0]][0].cpu().detach().numpy()
        attention_weights_decoder_cross      = attention_weights[attention_keys[-1]][0].cpu().detach().numpy()

        if USE_WANDB:
            wandb.log({
                "train_loss"       : train_loss,
                "train_perplexity" : train_perplexity,
                "learning_rate"    : curr_lr,
            })


        save_attention_plot(str(attn_img_root), attention_weights_decoder_cross, epoch,    mode='dec_lm')
        save_attention_plot(str(attn_img_root), attention_weights_decoder_self, epoch+100, mode='dec_lm')
        if config["scheduler"] == "ReduceLR":
            scheduler.step(train_perplexity)
        else:
            scheduler.step()

        ### Highly Recommended: Save checkpoint in drive and/or wandb if accuracy is better than your current best
        epoch_model_path = os.path.join(checkpoint_root, (checkpoint_last_epoch_filename + str(epoch) + '.pth'))
        save_model(model, optimizer, scheduler, {'perplexity': train_perplexity}, epoch, epoch_model_path)

        if best_perplexity >= train_perplexity:
            best_perplexity = train_perplexity
            save_model(model, optimizer, scheduler, {'perplexity': train_perplexity}, epoch, best_loss_model_path)
            print("Saved best perplexity model")

elif config["mode"] == "dec_cond_lm":
    epochs = config["pretrain_epochs"]
    for epoch in range(e, epochs):

        print("\nEpoch {}/{}".format(epoch+1, epochs))

        curr_lr = float(optimizer.param_groups[0]["lr"])

        train_loss, train_perplexity, attention_weights = train_step(
            model,
            criterion=loss_func,
            ctc_loss=None,
            ctc_weight=0.,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            train_loader=train_loader,
            tokenizer=Tokenizer,
            mode='dec_cond_lm',
            config=config
        )

        print("\nEpoch {}/{}: \nTrain Loss {:.04f}\t Train Perplexity {:.04f}\t Learning Rate {:.06f}".format(
            epoch + 1, epochs, train_loss, train_perplexity, curr_lr))


        levenshtein_distance, json_out, wer, cer = validate_step(
            model,
            val_loader=val_loader,
            tokenizer=Tokenizer,
            device=device,
            mode='dec_cond_lm',
            threshold=5
        )


        fpath = os.path.join(text_root, f'dec_cond_lm_{epoch+1}_out.json')
        with open(fpath, "w") as f:
            json.dump(json_out, f, indent=4)

        print("Levenshtein Distance : {:.04f}".format(levenshtein_distance))
        print("WER                  : {:.04f}".format(wer))
        print("CER                  : {:.04f}".format(cer))

        attention_keys = list(attention_weights.keys())
        attention_weights_decoder_self   = attention_weights[attention_keys[0]][0].cpu().detach().numpy()
        attention_weights_decoder_cross  = attention_weights[attention_keys[-1]][0].cpu().detach().numpy()

        if USE_WANDB:
            wandb.log({
                "train_loss"       : train_loss,
                "train_perplexity" : train_perplexity,
                "learning_rate"    : curr_lr,
                "lev_dist"         : levenshtein_distance,
                "WER"              : wer,
                "CER"              : cer
            })


        save_attention_plot(str(attn_img_root), attention_weights_decoder_cross, epoch,     mode='dec_cond_lm')
        save_attention_plot(str(attn_img_root), attention_weights_decoder_self,  epoch+100, mode='dec_cond_lm')
        if config["scheduler"] == "ReduceLR":
            scheduler.step(levenshtein_distance)
        else:
            scheduler.step()

        ### Highly Recommended: Save checkpoint in drive and/or wandb if accuracy is better than your current best
        epoch_model_path = os.path.join(checkpoint_root, (checkpoint_last_epoch_filename + str(epoch) + '.pth'))
        save_model(model, optimizer, scheduler, {'perplexity': train_perplexity, 'CER': cer}, epoch, epoch_model_path)

        if best_cer > cer:
            best_loss = train_loss
            best_cer = cer
            save_model(model, optimizer, scheduler, {'perplexity': train_perplexity, 'CER': cer}, epoch, best_loss_model_path)
            print("Saved best CER model")

elif config["mode"] == "full":
    # freeze 
    if e <= config["pretrain_epochs"]:
        if config["pre_mode"] == "dec_cond_lm":
            for name, param in model.named_parameters():
                if name.startswith("embedding"):
                    param.requires_grad = False
        """
        # freeze encoder
        for name, param in model.named_parameters():
            if name.startswith("encoder"):
                param.requires_grad = False
        """
        if config["pre_mode"] in ("dec_lm", "dec_cond_lm"):
            # freeze decoder
            for name, param in model.named_parameters():
                if name.startswith("decoder"):
                    param.requires_grad = False
        """
        # freeze encoder
        for name, param in model.named_parameters():
            if name.startswith("encoder"):
                param.requires_grad = False
        """
    #unfreeze
    epochs = config["pretrain_epochs"] + config["train_epochs"]
    for epoch in range(e, epochs):
        print("\nEpoch {}/{}".format(epoch+1, epochs))

        if epoch - e == 4:
            # unfreeze embedding
            if config["pre_mode"] == "dec_cond_lm":
                print("Unfreezing embedding")
                for name, param in model.named_parameters():
                    if name.startswith("embedding"):
                        param.requires_grad = True
            """
            # unfreeze encoder
            for name, param in model.named_parameters():
                if name.startswith("encoder"):
                    param.requires_grad = True
            """
            # unfreeze decoder
            if config["pre_mode"] in ("dec_lm", "dec_cond_lm"):
                print("Unfreezing decoder")
                for name, param in model.named_parameters():
                    if name.startswith("decoder"):
                        param.requires_grad = True

        curr_lr = float(optimizer.param_groups[0]["lr"])

        train_loss, train_perplexity, attention_weights = train_step(
            model,
            criterion=loss_func,
            ctc_loss=ctc_loss_fn,
            ctc_weight=config['ctc_weight'],
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            train_loader=train_loader,
            tokenizer=Tokenizer,
            mode='full',
            config=config
        )

        print("\nEpoch {}/{}: \nTrain Loss {:.04f}\t Train Perplexity {:.04f}\t Learning Rate {:.06f}".format(
            epoch + 1, epochs, train_loss, train_perplexity, curr_lr))


        levenshtein_distance, json_out, wer, cer = validate_step(
            model,
            val_loader=val_loader,
            tokenizer=Tokenizer,
            device=device,
            mode='full',
            threshold=5
        )

        fpath = os.path.join(text_root, f'full_{epoch+1}_out.json')
        with open(fpath, "w") as f:
            json.dump(json_out, f, indent=4)

        print("Levenshtein Distance : {:.04f}".format(levenshtein_distance))
        print("WER                  : {:.04f}".format(wer))
        print("CER                  : {:.04f}".format(cer))

        attention_keys = list(attention_weights.keys())
        attention_weights_decoder_self   = attention_weights[attention_keys[0]][0].cpu().detach().numpy()
        attention_weights_decoder_cross  = attention_weights[attention_keys[-1]][0].cpu().detach().numpy()

        if USE_WANDB:
            wandb.log({
                "train_loss"       : train_loss,
                "train_perplexity" : train_perplexity,
                "learning_rate"    : curr_lr,
                "lev_dist"         : levenshtein_distance,
                "WER"              : wer,
                "CER"              : cer
            })
        save_attention_plot(str(attn_img_root), attention_weights_decoder_cross, epoch,     mode='full')
        save_attention_plot(str(attn_img_root), attention_weights_decoder_self, epoch+100,  mode='full')
        if config["scheduler"] == "ReduceLR":
            scheduler.step(levenshtein_distance)
        
        # unfreeze decoder
        for name, param in model.named_parameters():
            if name.startswith("decoder"):
                param.requires_grad = True
            epoch_model_path = os.path.join(checkpoint_root, (checkpoint_last_epoch_filename + str(epoch) + '-2'+'.pth'))
            save_model(model, optimizer, scheduler, {'perplexity': train_perplexity, 'CER': cer}, epoch, epoch_model_path)

            if best_cer > cer:
                best_loss = train_loss
                best_cer = cer
                save_model(model, optimizer, scheduler, {'perplexity': train_perplexity, 'CER': cer}, epoch, best_loss_model_path)
                print("Saved best distance model")
else:
    print("Predict only")
### Finish your wandb run
if USE_WANDB:
    run.finish()
#### ----------------------------------------------------------------------------------------------------------------------

    


predictions = test_step(
        model,
        test_loader=test_loader,
        tokenizer=Tokenizer,
        device=device,
        config=config
)

#save to csv
import csv
csv_file_path = "submission.csv"

# Write the list to the CSV with index as the first column
with open(csv_file_path, mode="w", newline="") as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(["Index", "Labels"])
    # Write list items with index
    for idx, item in enumerate(predictions):
        writer.writerow([idx, item])

print(f"CSV file saved to {csv_file_path}")





