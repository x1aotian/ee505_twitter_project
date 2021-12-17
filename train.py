import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
from sklearn.metrics import f1_score
from tqdm import trange

def L2Loss(model,alpha=0.0000007):
    l2_loss = torch.tensor(0.0,requires_grad = True)
    for name,parma in model.named_parameters():
        if 'bias' not in name:
            l2_loss = l2_loss + (0.5*alpha * torch.sum(torch.pow(parma,2)))
    return l2_loss

def train(train_iter, dev_iter, model, args):
    if args.cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_acc = 0

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    f = open(args.save_dir+"/loss.csv", 'w')
    f.write("epoch,   loss,   evl_loss,   f1,   evl_f1")

    for epoch in range(1, args.epochs+1):
        print('----------------------')
        print('Epoch: {}/{}'.format(epoch, args.epochs))
        avg_loss = 0
        step = 0
        batch_num = len(train_iter.dataset)//train_iter.batch_size + 1
        avg_f1 = 0
        for batch in train_iter:
            step += 1
            model.train()
            feature, target = batch.text, batch.label
            feature.t_() # batch first, index align
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            optimizer.zero_grad()
            logit = model(feature)
            loss = F.cross_entropy(logit, target) + L2Loss(model)
            loss.backward()
            optimizer.step()
            avg_f1 += f1_score(torch.max(logit, 1)[1].view(target.size()).data.cpu(), target.data.cpu())

            avg_loss += loss.item()
            corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
            accuracy = 100.0 * corrects/batch.batch_size
            sys.stdout.write(
                '\rBatch[{}/{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(step, batch_num,
                                                                         loss.item(),
                                                                         accuracy.item(),
                                                                         corrects.item(),
                                                                         batch.batch_size))

        avg_loss /= len(train_iter.dataset)
        avg_loss *= train_iter.batch_size
        avg_f1 /= batch_num
        print('\nTraining - avg_loss: {:.6f}  avg_f1:  {:.6f}'.format(avg_loss, avg_f1))

        dev_avg_loss, dev_acc, evl_f1 = eval(dev_iter, model, args)
        if dev_acc > best_acc:
            best_acc = dev_acc
            if args.save_best:
                save(model, args.save_dir, 'best', epoch)

        if epoch % args.save_interval == 0:
            save(model, args.save_dir, 'snapshot', epoch)

        f.write('{}, {:.6f}, {:.6f}, {:.6f}, {:6f}\n'.format(epoch, avg_loss, dev_avg_loss, avg_f1, evl_f1))

    f.close()


def eval(data_iter, model, args):
    model.eval()
    corrects, avg_loss = 0, 0

    step = 0
    batch_num = len(data_iter.dataset)//data_iter.batch_size + 1
    correct_num = 0
    f1_avg = 0
    # if args.cuda:
        # f1_avg.cuda()

    for batch in data_iter:
        step += 1
        feature, target = batch.text, batch.label
        feature.t_()  # batch first, index align
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        logit = model(feature)
        loss = F.cross_entropy(logit, target, reduction='sum') + L2Loss(model)

        avg_loss += loss.item()
        corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
        correct_num += corrects

        f1_avg += f1_score(torch.max(logit, 1)[1].view(target.size()).data.cpu(), target.data.cpu())

        accuracy = 100.0 * corrects / batch.batch_size
        sys.stdout.write(
            '\rBatch[{}/{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(step, batch_num,
                                                                     loss.item(),
                                                                     accuracy.item(),
                                                                     corrects.item(),
                                                                     batch.batch_size))

    avg_loss /= len(data_iter.dataset)
    f1_avg /= batch_num
    accuracy = correct_num/len(data_iter.dataset)*100
    print('\nEvaluation - avg_loss: {:.6f}  acc: {:.4f}%({}/{})  F1_avg: {:.4f}'.format(avg_loss,
                                                                                        accuracy,
                                                                                        correct_num,
                                                                                        len(data_iter.dataset),
                                                                                        f1_avg))
    return avg_loss, accuracy, f1_avg


def predict(text, model, text_field, label_feild, cuda_flag):
    assert isinstance(text, str)
    model.eval()
    # text = text_field.tokenize(text)
    text = text_field.preprocess(text)
    text = [[text_field.vocab.stoi[x] for x in text]]
    x = torch.tensor(text)
    x = autograd.Variable(x)
    if cuda_flag:
        x = x.cuda()
    print(x)
    output = model(x)
    _, predicted = torch.max(output, 1)
    return label_feild.vocab.itos[predicted.item()+1]


def kaggle_test(tweets, model, text_field, label_field, cuda_flag):
    model.eval()
    results = []
    tweets_vec = [[text_field.vocab.stoi[w] for w in l.split(" ")] for l in tweets]
    tweets_vec = [i+[0]*(30-len(i)) for i in tweets_vec]
    for i in trange(len(tweets_vec)):
        veci = tweets_vec[i]
        x = torch.tensor([veci])
        x = autograd.Variable(x)
        if cuda_flag:
            x = x.cuda()
        output = model(x)
        _, predicted = torch.max(output, 1)
        results.append(int(predicted))

    return results

def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_epoch_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)
