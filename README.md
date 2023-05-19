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

