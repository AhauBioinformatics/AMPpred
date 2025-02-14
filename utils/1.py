import os
import pandas as pd
import numpy as np
import torch
import re
torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

from sklearn.metrics import accuracy_score, precision_recall_fscore_support,matthews_corrcoef,roc_auc_score
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification, AdamW,BertForSequenceClassification,BertModel
from tensorflow import keras
# load appropriate tokenizer and fine-tuned model
tokenizer = AutoTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False)
model = BertForSequenceClassification.from_pretrained("model/fold1")

with open('test.csv') as l:
  with open('7.16pred.txt','w') as r:
    with open('7.16prob.txt','w') as f:
      lines=l.readlines()
      for line in lines:
        input_seq=line.split(',')[1]
        input_seq_spaced = ' '.join([ input_seq[i:i+1] for i in range(0, len(input_seq), 1) ])
        input_seq_spaced = re.sub(r'[UZOB]', 'X', input_seq_spaced)
        input_seq_tok = tokenizer(input_seq_spaced, return_tensors = 'pt')
        output = model(**input_seq_tok)
        logits = output[0]
        y_prob = torch.sigmoid(logits)[:,1].detach().numpy()
        f.write(str(y_prob)[1:-1])
        f.write('\n')
        y_pred = y_prob > 0.5
        if y_pred == True:
          input_class ='1'
        else:
          input_class ='0'
        r.write(input_class)
        r.write('\n')
