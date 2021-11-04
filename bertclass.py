# Tosin Adewumi
"""
Original codes from Coursera
"""

import argparse
import torch
from tqdm import tqdm, trange
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
import random
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords

from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from transformers import BertForSequenceClassification
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup

parser = argparse.ArgumentParser(description='PyTorch Transformer Language Model')
parser.add_argument('--cuda', default='cuda', action='store_true', help='use CUDA')
parser.add_argument('--ofile1', type=str, default='out_new13idiomsbert.txt', help='output file')
args = parser.parse_args()

# constants
batch_size = 64
epochs = 7
seed_val = 17
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

class MakeSentence(object):
    """
    Makes sentences of the data of tokens passed to it
    """
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w.lower(), p, t) for w, p, t in zip(s["token"].values.tolist(), s["pos"].values.tolist(), s["class"].values.tolist())]
        self.grouped = self.data.groupby("id").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except ValueError:
            return None

def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average=None), f1_score(labels_flat, preds_flat, average="weighted")

def accuracy_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return accuracy_score(labels_flat, preds_flat, normalize='False')
    
def confusion_matrix_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    print(confusion_matrix(labels_flat, preds_flat))

def accuracy_per_class(preds, labels):
    label_dict_inverse = {v: k for k, v in label_dict.items()}

    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat==label]
        y_true = labels_flat[labels_flat==label]
        print(f'Class: {label_dict_inverse[label]}')
        print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n')
        with open(args.ofile1, "a+") as f:
            s = f.write(f'Class: {label_dict_inverse[label]}' + "\n" + f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n' + "\n")

def evaluate(dataloader_val):
    model.eval()
        
    loss_val_total = 0
    predictions, true_vals = [], []
        
    for batch in dataloader_val:
            
        batch = tuple(b.to(device) for b in batch)
            
        inputs = {'input_ids':      batch[0],
                    'attention_mask': batch[1],
                    'labels':         batch[2],
                    }

        with torch.no_grad():        
                outputs = model(**inputs)
                
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
        
    loss_val_avg = loss_val_total/len(dataloader_val) 
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)

    return loss_val_avg, predictions, true_vals

def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = BeautifulSoup(text).text # HTML decoding
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub('', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    #text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text
    return text

if __name__ == "__main__":
    device = torch.device(args.cuda)        # initialize device
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
            device = torch.device("cuda" if args.cuda else "cpu")
    else:
        device = torch.device("cpu")
    #df = pd.read_csv('title_conference.csv')
    labelclass = []
    df = pd.read_csv("corpus/idiomscorpus.csv", encoding="latin1").fillna(method="ffill")
    get_sent = MakeSentence(df)                                               # instantiate sentence maker
    sentences = [" ".join(s[0] for s in sent) for sent in get_sent.sentences]   # concat data (originally in tokens) into sentences
    labels = [[s[2] for s in sent] for sent in get_sent.sentences]      # construct true labels for each sentence
    #b = [x for l in labels for x in l]                                 # flatten into 1 list NOT list of lists
    for numb in range(len(labels)):
        labelclass.append(labels[numb][0])
    corp_dict = {'sentence':sentences, 'labelclass':labelclass}                  # construct dict from the 2 lists
    df = pd.DataFrame(corp_dict)
    #print(df['labelclass'].value_counts())                          # print the no of samples/category
    df['sentence'] = df['sentence'].apply(clean_text)                           # pre-processing
    possible_labels = df.labelclass.unique()
    label_dict = {}
    for index, possible_label in enumerate(possible_labels):
        label_dict[possible_label] = index
    print(label_dict)
    df['label'] = df.labelclass.replace(label_dict)                 # replace labels with their nos

    # Stratified dataset split due to data imbalance
    X_train, X_val, y_train, y_val = train_test_split(df.index.values,
                                                  df.label.values,
                                                  test_size=0.15,
                                                  random_state=42,
                                                  stratify=df.label.values)

    df['data_type'] = ['not_set']*df.shape[0]
    df.loc[X_train, 'data_type'] = 'train'
    df.loc[X_val, 'data_type'] = 'val'
    #print(df.groupby(['labelclass', 'label', 'data_type']).count())

    # load tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    encoded_data_train = tokenizer.batch_encode_plus(
        df[df.data_type=='train'].sentence.values, 
        add_special_tokens=True, 
        return_attention_mask=True,
        padding='max_length',
        max_length=256, 
        return_tensors='pt'
    )

    encoded_data_val = tokenizer.batch_encode_plus(
        df[df.data_type=='val'].sentence.values, 
        add_special_tokens=True, 
        return_attention_mask=True,
        padding='max_length',
        max_length=256, 
        return_tensors='pt'
    )

    input_ids_train = encoded_data_train['input_ids']
    attention_masks_train = encoded_data_train['attention_mask']
    labels_train = torch.tensor(df[df.data_type=='train'].label.values)

    input_ids_val = encoded_data_val['input_ids']
    attention_masks_val = encoded_data_val['attention_mask']
    labels_val = torch.tensor(df[df.data_type=='val'].label.values)

    dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
    dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)

    # instantiate the model
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=len(label_dict),
                                                      output_attentions=False,
                                                      output_hidden_states=False)
    model = model.to(device)

    dataloader_train = DataLoader(dataset_train, 
                                sampler=RandomSampler(dataset_train), 
                                batch_size=batch_size)

    dataloader_validation = DataLoader(dataset_val, 
                                    sampler=SequentialSampler(dataset_val), 
                                    batch_size=batch_size)

    optimizer = AdamW(model.parameters(), lr=1e-5, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=0,
                                            num_training_steps=len(dataloader_train)*epochs)

# training loop
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    with open(args.ofile1, "a+") as f:
        s = f.write("\n" + "\n")
    print("Training.... ")
    for epoch in tqdm(range(1, epochs+1)):
        model.train()
        loss_train_total = 0

        progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
        for batch in progress_bar:

            model.zero_grad()
            
            batch = tuple(b.to(device) for b in batch)
            
            inputs = {'input_ids':      batch[0],
                    'attention_mask': batch[1],
                    'labels':         batch[2],
                    }       

            outputs = model(**inputs)
            
            loss = outputs[0]
            loss_train_total += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            
            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})
        print("Saving the model.... ")
        torch.save(model.state_dict(), f'data_volume/finetuned_BERT_epoch_{epoch}.model')
        tqdm.write(f'\nEpoch {epoch}')
        loss_train_avg = loss_train_total/len(dataloader_train)
        tqdm.write(f'Training loss: {loss_train_avg}')

        val_loss, predictions, true_vals = evaluate(dataloader_validation)
        val_f1, val_f1_w = f1_score_func(predictions, true_vals)
        confusion_matrix_func(predictions, true_vals)
        tqdm.write(f'Validation loss: {val_loss}')
        tqdm.write(f'F1 Score (Classes): {val_f1}' + f'F1 Score (Weighted): {val_f1_w}')
        print('Sci accuracy:', accuracy_score_func(predictions, true_vals))
        accuracy_per_class(predictions, true_vals)
        with open(args.ofile1, "a+") as f:
            s = f.write(f'Training loss: {loss_train_avg}' + "\n" + f'Validation loss: {val_loss}' + "\n" + f'F1 Score (Classes): {val_f1}' + f'F1 Score (Weighted): {val_f1_w}' + "\n")

# # evaluating the model
# print("Evaluation.... ")
# model.to(device)
# model.load_state_dict(torch.load('data_volume/finetuned_BERT_epoch_1.model', map_location=torch.device('cpu')))

# _, predictions, true_vals = evaluate(dataloader_validation)
# accuracy_per_class(predictions, true_vals)
# #confusion_matrix_func(predictions, true_vals)
# #accuracy_score(true_vals, predictions)
