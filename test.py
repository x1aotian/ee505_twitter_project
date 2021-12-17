import pandas as pd
from sklearn.model_selection import train_test_split
import spacy
import torch
from torchtext.legacy import data, datasets
from torchtext.vocab import Vectors
from torch.nn import init


header = ['target', 'tweet_index', 'time stamp', '', 'user', 'tweet']
data = pd.read_csv('data/ori/twitter_sentiments_data.csv', sep=',', names=header,engine='python',error_bad_lines=False)
data = data.drop(['tweet_index', 'time stamp', '', 'user'], axis=1)
data = data.drop(0)

train, val = train_test_split(data, test_size=0.2, random_state=0, shuffle=True)
train.to_csv("train.csv", index=False)
val.to_csv("val.csv", index=False)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

spacy_en = spacy.load('en_core_web_sm')

def tokenizer(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

LABEL = data.Field(sequential=False, use_vocab=False)
TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True)

train,val = data.TabularDataset.splits(
        path='.', train='train.csv', validation='val.csv', format='csv',skip_header=True,
        fields=[('target', LABEL), ('tweet', TEXT)])

TEXT.build_vocab(train, vectors='glove.6B.100d', max_size=30000)
TEXT.vocab.vectors.unk_init = init.xavier_uniform

train_iter = data.BucketIterator(train, batch_size=128, shuffle=True, device=DEVICE)

val_iter = data.BucketIterator(val, batch_size=128, shuffle=True, device=DEVICE)

print("end")