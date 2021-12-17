#! /usr/bin/env python
import os
import argparse
import datetime
import torch
import torchtext.legacy.data as data
import torchtext.datasets as datasets
import model
import train
from torch.nn import init
import spacy
import pandas as pd
from tqdm import trange
from preprocess import preprocess_text

from preprocess import preprocess

parser = argparse.ArgumentParser(description='CNN text classificer')
# learning
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=12, help='number of epochs for train [default: 256]')
parser.add_argument('-batch_size', type=int, default=64, help='batch size for training [default: 64]')
parser.add_argument('-log_interval', type=int, default=1,
                    help='how many step of samples to wait before logging training status [default: 1]')
parser.add_argument('-test_interval', type=int, default=1,
                    help='how many step of batches to wait before testing [default: 100]')
parser.add_argument('-save_interval', type=int, default=5, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save_dir', type=str, default='results', help='where to save the results')
parser.add_argument('-early_stop', type=int, default=1000,
                    help='iteration numbers to stop without performance increasing')
parser.add_argument('-save_best', type=bool, default=True, help='whether to save when get best performance')
# data
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
# model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max_norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embed_dim', type=int, default=128, help='number of embedding dimension [default: 128]')
parser.add_argument('-kernel_num', type=int, default=100, help='number of each kind of kernel')
parser.add_argument('-kernel_sizes', type=str, default='3,4,5',
                    help='comma-separated kernel size to use for convolution')
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
# device
# parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no_cuda', action='store_true', default=False, help='disable the gpu')
# option
# parser.add_argument('-snapshot', type=str, default='results/1216_1/best_epoch_6.pt', help='filename of model snapshot [default: None]')
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')

parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
parser.add_argument('-test', action='store_true', default=False, help='train or test')

parser.add_argument('-ori_csv', type=str, default='data/twitter_sentiments_data.csv', help='method of vectorize')
parser.add_argument('-first', action='store_true', default=False, help='first time needs to split dataset and do preprocess')
parser.add_argument('-vector', type=str, default='glove.6B.100d', help='path of original csv file')
parser.add_argument('-num_class', type=int, default=2, help='number of classes, 2 or 3')
parser.add_argument('-kaggle_test', action='store_true', default=False, help='train or test')

args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device("cpu")

if args.first:
    print("Data preprocessing ...")
    preprocess(args.ori_csv)

print("Loading data...")

label_field = data.Field(sequential=False, use_vocab=False)
text_field = data.Field(sequential=True, lower=True)

train_set, val_set = data.TabularDataset.splits(
    path='.', train='data/train_set.csv', validation='data/val_set.csv', format='csv', skip_header=True,
    fields=[('label', label_field), ('text', text_field)])

text_field.build_vocab(train_set, vectors=args.vector, max_size=60000)
text_field.vocab.vectors.unk_init = init.xavier_uniform

train_iter = data.BucketIterator(train_set, batch_size=args.batch_size, shuffle=True, device=device)
val_iter = data.BucketIterator(val_set, batch_size=args.batch_size, shuffle=False, device=device)

# update args and print
args.embed_num = len(text_field.vocab)
args.class_num = args.num_class
args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))

# model
cnn = model.CNN_Text(args)
if args.snapshot is not None:
    print('\nLoading model from {}...'.format(args.snapshot))
    cnn.load_state_dict(torch.load(args.snapshot, map_location=torch.device('cpu')))

if args.cuda:
    torch.cuda.set_device(0)
    cnn = cnn.cuda()

# train or predict
if args.predict is not None:
    label = train.predict(args.predict, cnn, text_field, label_field, args.cuda)
    print('\n[Text]  {}\n[Label] {}\n'.format(args.predict, label))

elif args.kaggle_test:
    header = ['tweet_index', 'time stamp', '', 'user', 'tweet']
    data = pd.read_csv('data/twitter_sentiments_evaluation.csv', sep=',', names=header, engine='python', error_bad_lines=False)
    # data = pd.read_csv('data/test.csv', sep=',', names=header, engine='python', error_bad_lines=False)
    data = data.drop(0)

    predict_tweets = []
    for i in trange(1, len(data)+1):
        predict_tweets.append(preprocess_text(data.loc[i, 'tweet']))

    predicts = train.kaggle_test(predict_tweets, cnn, text_field, label_field, args.cuda)
    # data.insert(0, 'target', predicts)
    # data.to_csv('results/eval_1216_1.csv', index=False)
    pred_pd = pd.DataFrame()
    pred_pd['tweet_index'] =data['tweet_index'].copy()
    pred_pd['target'] = predicts
    pred_pd.to_csv('results/eval_1216_1.csv', index=False)
    print("Predict end")

else:
    print()
    try:
        train.train(train_iter, val_iter, cnn, args)
    except KeyboardInterrupt:
        print('\n' + '-' * 89)
        print('Exiting from training early')

