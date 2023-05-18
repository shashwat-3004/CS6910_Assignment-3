
# Library Import

import wandb
import pandas as pd      
import numpy as np
import torch
import torch.nn as nn
import random
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from zipfile import ZipFile
import argparse

############################
# To unzip the aksharantar_sample.zip#
#######################################

'''
zip_path="C:\\Users\\ASUS\\Desktop\\DL-CS6910\\Assignment-3\\aksharantar_sampled.zip"

with ZipFile(zip_path, 'r') as zObject:
    zObject.extractall(
        path="C:\\Users\\ASUS\\Desktop\\DL-CS6910\\Assignment-3")
zObject.close()    
'''

## To get training, test and val data
def get_data(lang):
    train_csv=f"C:\\Users\\ASUS\\Desktop\\DL-CS6910\\Assignment-3\\aksharantar_sampled\\{lang}\\{lang}_train.csv"
    test_csv=f"C:\\Users\\ASUS\\Desktop\\DL-CS6910\\Assignment-3\\aksharantar_sampled\\{lang}\\{lang}_test.csv"
    val_csv=f"C:\\Users\\ASUS\\Desktop\\DL-CS6910\\Assignment-3\\aksharantar_sampled\\{lang}\\{lang}_valid.csv"
    
    return train_csv, test_csv, val_csv

## Function to get maximum length of sequence and token map
def get_token_maps_length(data):   # for train_csv only
    
    input_texts=[]
    target_texts=[]
    
    df=pd.read_csv(data,header=None, names=["1","2"]).astype(str)  # read the csv file

    # all the input and target texts with start sequence and end sequence added to target 
    for index, row in df.iterrows():
        input_text = row['1']
        target_text = row['2']
        if target_text == '' or input_text == '':
            continue
        target_text = "\t" + target_text + "\n"  # EOS - \n & SOS - \t
        input_texts.append(input_text)
        target_texts.append(target_text)
    
    english_tokens=set()
    hindi_tokens = set()

    for x,y in zip(input_texts,target_texts):
        for ch in x:
            english_tokens.add(ch)              # english and hindi tokens added
        for ch in y:
            hindi_tokens.add(ch)
    
    english_tokens = sorted(list(english_tokens))
    hindi_tokens = sorted(list(hindi_tokens))

    eng_token_map = dict([(ch,i+1) for i,ch in enumerate(english_tokens)])       # dict for english and hindi tokens
    hin_token_map = dict([(ch,i+1) for i,ch in enumerate(hindi_tokens)])

    eng_token_map["<UNK>"]=len(english_tokens)+1
    hin_token_map["<UNK>"]=len(hindi_tokens)+1
    eng_token_map['<PAD>']=0
    hin_token_map['<PAD>']=0

    reverse_eng_map = dict([(i,char) for char,i in eng_token_map.items()])
    reverse_hin_map = dict([(i,char) for char,i in hin_token_map.items()])

    max_eng_len = max([len(i) for i in input_texts])    # Max sequence length for eng and hindi word
    max_hin_len = max([len(i) for i in target_texts])

    return eng_token_map,hin_token_map,reverse_eng_map,reverse_hin_map,max_eng_len,max_hin_len


def pre_process(data,eng_token_map,hin_token_map,max_eng_len,max_hin_len):
    '''
    Parameters:

        data: train/val/test csv

        eng_token_map: dictionary of english characters

        hin_token_map: dictionary of hindi characters

        max_eng_len: max sequence length of english word

        max_hin_len: max sequence length of hindi word
    
    Returns:

        a: embedding of english words
        b: embedding of hindi words
    
    '''
    
    input_texts = []
    target_texts = []
    
    df = pd.read_csv(data, header=None, names=["1", "2"]).astype(str)

    for index, row in df.iterrows():
      input_text = row['1']
      target_text = row['2']
      if target_text == '' or input_text == '':
          continue
      target_text = "\t" + target_text + "\n"
      input_texts.append(input_text)
      target_texts.append(target_text)

    
    a = np.zeros((len(input_texts),max_eng_len+2),dtype="float32")
    b = np.zeros((len(target_texts),max_hin_len+2),dtype="float32")
    
    
    for i,(x,y) in enumerate(zip(input_texts,target_texts)):
        for j,ch in enumerate(x):
            a[i,j] = eng_token_map.get(ch,eng_token_map["<UNK>"])     # Adding unknown token as well

        for j,ch in enumerate(y):
            b[i,j] = hin_token_map.get(ch,hin_token_map["<UNK>"])
        
      
    return a,b


class CustomDataset(Dataset):                      # For dataloader in pytorch
    def __init__(self, input_data, target_data):
        self.input_data = input_data
        self.target_data = target_data
    

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        input_seq = self.input_data[idx]
        target_seq = self.target_data[idx]

        return input_seq, target_seq               # returns the sequneces in torch.tensor format

def custom_collate(batch):
    input_seqs, target_seqs= zip(*batch)
    input_seqs = torch.from_numpy(np.stack(input_seqs, axis=1))            # In general, dataloader has batch size is 1st dimension,
                                                                           # this function changes the batch size to 2nd dimension
    target_seqs = torch.from_numpy(np.stack(target_seqs, axis=1))

    return input_seqs, target_seqs
###############################################
# Processing complete
######################################################################

# Seq2Seq Model

class Encoder(nn.Module):               # Encoder class
    def __init__(self, input_size, embed_dim, hidden_size, num_layers, dropout,cell_type):
        '''
        Parameters:

            input_size: input sequnece length 

            embed_dim: Embedding dimension size

            hidden_size: Hidden unit size

            Num layers: Number of layers of encoder

            Dropout: dropout

            cell_type: LSTM/GRU/RNN

        '''

        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, embed_dim,padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.cell_type=cell_type

        if cell_type=="LSTM":
          self.rnn = nn.LSTM(embed_dim, hidden_size, num_layers, dropout=dropout)
        elif cell_type=="GRU":
          self.rnn=nn.GRU(embed_dim,hidden_size,num_layers,dropout=dropout)
        else:
          self.rnn=nn.RNN(embed_dim,hidden_size,num_layers,dropout=dropout)
    
    def forward(self, x):
        embedded = self.dropout(self.embedding(x))

        if self.cell_type=="LSTM":
          output, (hidden, cell) = self.rnn(embedded)
          return hidden, cell
        
        elif self.cell_type=="GRU":
          output, hidden = self.rnn(embedded)

          return output, hidden
        
        else:
          output, hidden = self.rnn(embedded)

          return output,hidden


class Decoder(nn.Module):                 # Decoder Class
    def __init__(self, output_size, embed_dim, hidden_size, num_layers, dropout,cell_type):
        '''
        Parameters:

        output_size: target seq length

        embed_dim: Embedding dimension size

        hidden_size: Hidden unit size

        num_layers: No of decoder ;ayers

        dropout: Dropour

        cell_type: RNN/LSTM/GRU
        
        '''


        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.cell_type=cell_type
        self.embedding = nn.Embedding(output_size, embed_dim,padding_idx=0)
        if cell_type=="LSTM":
          self.rnn = nn.LSTM(embed_dim, hidden_size, num_layers,  dropout=dropout)
        elif cell_type=="GRU":
          self.rnn=nn.GRU(embed_dim,hidden_size,num_layers,dropout=dropout)
        else:
          self.rnn=nn.RNN(embed_dim,hidden_size,num_layers,dropout=dropout)

        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout=nn.Dropout(dropout)

    def forward(self, x, hidden, cell):
        x = x.unsqueeze(0)
        embedded = self.dropout(self.embedding(x))
        if self.cell_type=="LSTM":
          output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
          output= self.fc(output)
          output = output.squeeze(0)
          return output, hidden, cell
        
        elif self.cell_type=="GRU":
          output, hidden=self.rnn(embedded,hidden)
          output=self.fc(output)
          output=output.squeeze(0)
          return output, hidden
        
        else:
          output, hidden=self.rnn(embedded,hidden)
          output=self.fc(output)
          output = output.squeeze(0)
          return output, hidden

class Seq2Seq(nn.Module):     # Seq2Seq model
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
       
    def forward(self, source, target, teacher_forcing_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(hin_token_map)

        outputs = torch.zeros(target_len,batch_size, target_vocab_size).to(device)
        if self.encoder.cell_type=="LSTM":
          hidden, cell = self.encoder(source)

          x = target[0]

          for t in range(1, target_len):
              output, hidden, cell = self.decoder(x, hidden, cell)
              outputs[t] = output
              top1 = output.argmax(1)
              if random.random() < teacher_forcing_ratio:
                  x = target[t]
              else:
                  x = top1

          return outputs
        
        elif self.encoder.cell_type=="GRU":
          enc_output,hidden = self.encoder(source)

          x = target[0]

          for t in range(1, target_len):
              output,hidden=self.decoder(x,enc_output,hidden,None)
              outputs[t] = output
              top1= output.argmax(1)
              if random.random() < teacher_forcing_ratio:          # tecacher forcing ratio 0f 0.5 has been used for training
                  x = target[t]
              else:
                  x = top1
          return outputs
        
        else:
          enc_output,hidden = self.encoder(source)

          x = target[0]

          for t in range(1, target_len):
              output,hidden=self.decoder(x,enc_output,hidden,None)
              outputs[t] = output
              top1= output.argmax(1)
              if random.random() < teacher_forcing_ratio:
                  x = target[t]
              else:
                  x = top1
          return outputs

# Model building
def build_model(cell = "LSTM",nunits = 64, enc_dec_layers = 2,embed_dim = 128,dropout=0):
    encoder = Encoder(input_size=len(eng_token_map), embed_dim=embed_dim, hidden_size=nunits, num_layers=enc_dec_layers, dropout=dropout,cell_type=cell)
    decoder = Decoder(output_size=len(hin_token_map), embed_dim=embed_dim, hidden_size=nunits, num_layers=enc_dec_layers, dropout=dropout,cell_type=cell)
    model = Seq2Seq(encoder, decoder)
    return model

# Function to train the model
def train(model, dataloader, criterion, optimizer, device):
    '''
    Parameters:

    model: seq2seq model

    dataloader: data load from here

    criterion: Cross-entropy

    optimizer: Adam
    
    device: CUDA/cpu
    
    '''
    model.train()
    total_loss = 0.0
    total_chars = 0
    correct_chars = 0
    for i, (input_seq, target_seq) in enumerate(dataloader):
        input_seq = input_seq.long().to(device)
        target_seq = target_seq.long().to(device)

        optimizer.zero_grad()

        output = model(input_seq, target_seq)
        _, predicted = torch.max(output, dim=2)


        for j in range(predicted.shape[1]):
              predicted_seq = predicted[:, j]
              targets_seq = target_seq[:, j]

              # Find the index of the first EOS token in the sequence (for character & word-level accuracy)
              eos_idx = (targets_seq == hin_token_map["\n"]).nonzero()
              if eos_idx.numel() > 0:
                  eos_idx = eos_idx[0][0]
                  predicted_seq = predicted_seq[:eos_idx]
                  targets_seq = targets_seq[:eos_idx]

              
        
        # reshape for cross-entropy loss
        output_flatten = output[1:].view(-1, output.shape[-1])
        trg_flatten = target_seq[1:].view(-1)

        loss = criterion(output_flatten, trg_flatten)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        predicted_seq=predicted_seq[1:].view(-1)
        targets_seq=targets_seq[1:].view(-1)
     
        correct_chars += torch.sum(predicted_seq == targets_seq).item()
        total_chars += targets_seq.numel()


    return total_loss / len(dataloader), correct_chars/total_chars           # Loss and accuracy

def evaluate(model, dataloader, criterion, device):
    '''
    Parameters:

    model: seq2seq model

    dataloader: to load data

    critetion: cross entropy loss

    device: CPU/CUDA
    
    '''
    model.eval()
    total_loss = 0.0

    total_chars = 0
    correct_chars = 0

    with torch.no_grad():
        for i, (input_seq, target_seq) in enumerate(dataloader):
            input_seq = input_seq.long().to(device)
            target_seq = target_seq.long().to(device)

            output = model(input_seq, target_seq,0)
            _, predicted = torch.max(output, dim=2)


            for j in range(predicted.shape[1]):
                predicted_seq = predicted[:, j]
                targets_seq = target_seq[:, j]

                # Find the index of the first EOS token in the sequence
                eos_idx = (targets_seq == hin_token_map["\n"]).nonzero()
                if eos_idx.numel() > 0:
                    eos_idx = eos_idx[0][0]
                    predicted_seq = predicted_seq[:eos_idx]
                    targets_seq = targets_seq[:eos_idx]


            # reshape for cross-entropy loss
            output_flatten = output[1:].view(-1, output.shape[-1])
            trg_flatten = target_seq[1:].view(-1)

            loss = criterion(output_flatten, trg_flatten)

            total_loss += loss.item()

            predicted_seq=predicted_seq[1:].view(-1)
            targets_seq=targets_seq[1:].view(-1)
     
            correct_chars += torch.sum(predicted_seq == targets_seq).item()
            total_chars += targets_seq.numel()

    return total_loss / len(dataloader), correct_chars/total_chars           




def main(args):
    
    global eng_token_map    # global variables
    global hin_token_map
    global device

    run = wandb.init(config=args)
    
    cell=wandb.config.cell
    hidden_units=wandb.config.hidden_units
    enc_dec_layers=wandb.config.enc_dec_layers   # wandb configs
    embed_dim=wandb.config.embed_dim
    lr=wandb.config.learning_rate
    epochs=wandb.config.epochs
    dropout=wandb.config.dropout
    batch_size=wandb.config.batch_size

    # Load data

    train_data,test_data,val_data=get_data("hin")   ## Hindi chosen as target data
    
    # get token maps and max sequence length

    eng_token_map,hin_token_map,reverse_eng_map,reverse_hin_map,max_eng_len,max_hin_len=get_token_maps_length(train_data)
    
    # Get embeddings 

    trainx, trainy = pre_process(train_data,eng_token_map,hin_token_map,max_eng_len,max_hin_len)
    valx, valy = pre_process(val_data,eng_token_map,hin_token_map,max_eng_len,max_hin_len)
    testx, testy = pre_process(test_data,eng_token_map,hin_token_map,max_eng_len,max_hin_len)

    # Train/Test/Val dataloader

    train_dataset = CustomDataset(trainx, trainy)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)

    val_dataset = CustomDataset(valx, valy)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)

    test_dataset = CustomDataset(testx, testy)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # run_name

    run_name = f"cell_{cell}_hunit_{hidden_units}_embed_dim_{embed_dim}_lr_{lr}_ep_{epochs}_enc_dec_layer_{enc_dec_layers}_dropout_{dropout}_bs_{batch_size}"
    print(run_name)

    #seq2seq model build

    model=build_model(cell = cell,nunits = hidden_units, enc_dec_layers =enc_dec_layers,embed_dim = embed_dim,dropout=dropout)

    model=model.to(device)
    
    # set criterion and optimizer

    criterion = nn.CrossEntropyLoss(ignore_index=hin_token_map["<PAD>"])
    optimizer = optim.Adam(model.parameters(),lr=lr)
    
    # model training and validation and logging it to wandb

    N_EPOCHS=epochs

    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS): 

        train_loss,train_acc = train(model, train_dataloader, criterion,optimizer,device)
        valid_loss,val_acc = evaluate(model, val_dataloader, criterion,device)
    
        wandb.log({"training_acc": train_acc, "validation_accuracy": val_acc, "training_loss": train_loss, "validation_loss": valid_loss, "Epoch": epoch+1})

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tTraining Accuracy: {:.6f} \tValidation Accuracy: {:.6f}'.format(
        epoch+1, train_loss, valid_loss, train_acc, val_acc))

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(),"model_p1.pt")
  
    wandb.run.name = run_name
    wandb.run.save()
    wandb.run.finish()

    model.load_state_dict(torch.load('model_p1.pt'))

    test_loss,test_acc=evaluate(model,test_dataloader,criterion,device)
    print(f'Test Loss: {test_loss:.9f}')
    print(f'Character Level Accuracy: {test_acc:.6f}')
    
'''    
    total_correct=0
    test_loader_batch_one = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate)

    model.eval()
    with torch.no_grad():
        for i, (input_seq,target_seq) in enumerate(test_loader_batch_one):
            input_seq = input_seq.long().to(device)
            target_seq = target_seq.long().to(device)

            outputs = model(input_seq,target_seq)
            output_idx = outputs[1:].squeeze(1).argmax(1)
            word_idx=[]
            real_word=[]
            for idx in output_idx.cpu():
                num=int(idx.numpy())
                if num == 2:
                    break
                else:
                    word_idx.append(reverse_hin_map[num])

            for idx in target_seq.cpu().numpy():
                if idx==2:
                    break
                elif idx==1:
                    pass
                else:
                    real_word.append(reverse_hin_map[int(idx)])
            print("".join(word_idx),"".join(real_word))
            if "".join(word_idx)=="".join(real_word):
                total_correct+=1

    print(f"Word-Level Accuracy(Exact string match):{total_correct/len(testy)}")
'''

if __name__=="__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-wp',"--wandb_project",type=str,default="Assignment-3")
    parser.add_argument("-we","--wandb_entity",type=str,default="shashwat_mm19b053")
    parser.add_argument("-c","--cell",type=str,default='LSTM',help="Cell Type")
    parser.add_argument("-e","--epochs",type=int,default=10,help='Number of epochs')
    parser.add_argument("-b","--batch_size",type=int,default=32,help='Batch Size')
    parser.add_argument("-lr","--learning_rate",type=float,default=0.0001,help="Learning Rate")
    parser.add_argument("-ed","--embed_dim",type=int,default=64,help="Embedding Dimension")
    parser.add_argument("-hd","--hidden_units",type=int,default=64,help="Hidden unit size")
    parser.add_argument("-er_dr","--enc_dec_layers",type=int,default=2,help="Number of encoder and decoder layer")
    parser.add_argument("-dp","--dropout",type=float,default=0.1)
    
args = parser.parse_args()


main(args)






   













