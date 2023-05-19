# CS6910_Assignment-3

# The repository consists code of Assignment based on transliteration from English to Hindi using seq2seq model with and without attention mechanism. 

The code has been written with pytorch, so its necessary to install pytorch. Wandb has been used for hyperparameter tuning and running sweeps.

Instructions to run ```train_simple.py``` & ```train_attention.py```  (Comments to code have been written here, .ipynb files are just to run codes on google collab)

- Go to the directory where the files are located.
- Run the following command, if you want to sync the run to the cloud: ```wandb online```
- Do python train.py ```--wandb_entity ENTITY_NAME --wandb_project PROJECT_NAME``` to run the script, where ```ENTITY_NAME``` & ```PROJECT_NAME``` is your entity name and proejct name. Currently, the default is set to mine.
- ```train.py``` can handle different arguments. The defaults are set to the hyperparameters which can be trained on my own personal PC GPU.
 Supported arguments:
 
 | Name | Default Value | Description |
| --- | ------------- | ----------- |
| `-wp`, `--wandb_project` | Assignment-1 | Project name used to track experiments in Weights & Biases dashboard |
| `-we`, `--wandb_entity` | shashwat_mm19b053  | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
| `-c`, `--cell` | LSTM | choices:  ["LSTM", "GRU","RNN"] |
| `-e`, `--epochs` | 10 |  Number of epochs to train neural network.|
| `-b`, `--batch_size` | 32 | Batch size used to train neural network. | 
| `-lr`, `--learning_rate` | 0.0001 | Learning rate used to optimize model parameters | 
| `-ed`, `--embed_dim` | 64 | Embedding Dimension | 
| `-hd`, `--hidden_units` | 64 | Hidden Unit Size|
| `-er_dr`, `--enc_dec_layers` | 2 | No. of encoder and decoder layer (Not used for ```train_attention.py```) |
| `-dp`, `--dropout` | 0.1 | dropout |

```train_simple/attention.py``` returns the wandb logs generated on Training and Validation dataset. It also prints out the loss associated with test dataset and character-level accuracy.

----------------------------------------------------------------
- ```pred_vanilla.csv``` has predictions of test dataset using simple seq2seq model
- ```pred_attention.csv``` has prediction of test dataset using seq2seq model with attention.
- ```Assignment_3.ipynb``` has wandb sweeps which I ran for simple seq2seq model
- ```Assignment_3_attention.ipynb``` has wandb sweeps which I ran for seqseq model with attention.
- ```Best_model_without_attention.ipynb``` has the best seq2seq model without attention
- ```Best_model_with_attention.ipynb``` has best seq2seq model with attention
----------------------------------------------------------------

Wandb Project Report Link: https://api.wandb.ai/links/shashwat_mm19b053/gag9wm0x

----------------------------------------------------------------

Functions and Classes Used:

- ```get_data```: Used to get training,test & validation data
- ```get_token_maps_length```: It is used to get english and hindi token map (dictionary) and maximum sequence length both in english and hindi.
- ```pre_process```: The function is used to get input embeddings using the token map for both english and hindi
- Class ```CustomDataset``` has been created for making a dataloader
- ```custom_collate```: It has been created so that batch size can be changed to 2nd dimension in dataloader instead of first.
- Encoder, Decoder, Seq2Seq class made for making the model.
- ```build_model```: Function that creates the seq2seq model.
- ```train```: To train the model
- ```evaluate```: To evaluate the model


### Workflow

The ```get_data``` function is first used to get the data in raw format, then ```get_token_maps_length``` function is used to get token maps as dictionary format for both english and hindi words, it is also used to get maximum sequence length for both english and hindi words. ```pre_process``` function is then used to get input embeddings using the token map. ```CustomDataset``` converts the data such that it can be used for pytorch.  This completes the processing of data part.

```build_model``` is then used to cretae the Seq2Seq model with different hyperparameters, ``` train``` is used for training and ```evaluate``` is used to evaluate on test and validation data.

## Results


 | | Without Attention | With Attention |
 |---------|-------|--------------|
 | Hidden Unit Size | 512 | 512 |
 | Embedding Dimension | 256 | 128 |
 | Epochs | 25 | 25 |
 | No of encoder-decoder | 2 | 1 |
 | Bi-directional | False | True(Encoder only) |
 | Learning Rate | 0.0001 | 0.0001 |
 | Batch Size | 64 | 64 |
 | Cell Type | LSTM | LSTM |
 | Dropout | 0.27 | 0.1 |
 | Test Accuracy(Exact String Match) | 35.57%| 40.72% |
