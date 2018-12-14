import torch, torchvision
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import numpy as np
import wandb

from dataset import SimDataset, collate_fn
from model import Net
from util import sec2hms
from tqdm import tqdm
import argparse
import time, os
st = time.time()

# Training settings
parser = argparse.ArgumentParser(description='Blind Source Separation using Top Down Attention')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 4)')
parser.add_argument('--valid-batch-size', type=int, default=8, metavar='N',
                    help='input batch size for testing (default: 2)')
parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=5e-3, metavar='LR',
                    help='learning rate (default: 1e-4)')
parser.add_argument('--subset', type=str, nargs='+', default='',
                    help='Specify subset condition `toy`')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=2, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-save-interval', type=int, default=24, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--valid-interval', type=int, default=12, metavar='N',
                    help='how many batches to wait before logging valid status')
parser.add_argument('--band', type=int, default=256,
                    help='specific subband of the model')
parser.add_argument('--bandwidth', type=int, default=1,
                    help='specific subband of the model')
parser.add_argument('--cache', action='store_true', default=False,
                    help='specific subband of the model')
parser.add_argument('--norm', action='store_true', default=False,
                    help='specific subband of the model')
parser.add_argument('--shuffle', action='store_true', default=False,
                    help='shuffle dataset')
parser.add_argument('--demix', action='store_true', default=False,
                    help='shuffle dataset')
parser.add_argument('--postfix', type=str, 
                    help='simple note for experiments')
args = parser.parse_args()

wandb.init(config=args)
args.cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

loader_kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}

def getdate():
    import time
    return time.strftime('%y%m%d')

savepath = os.path.join('result', '{}_{}'.format(getdate(), args.postfix))
if not os.path.exists(savepath):
    os.makedirs(savepath)
else:
    input("Path already exists, wish to continue?")
    os.system("rm -rf {}/*".format(savepath))

trainlosspath = os.path.join(savepath, 'train_loss_{}.log'.format(args.band))
validlosspath = os.path.join(savepath, 'valid_loss_{}.log'.format(args.band))
testlosspath  = os.path.join(savepath, 'test_loss_{}.log'.format(args.band))

bestmodelpath = os.path.join(savepath, 'model_best_{}.pt'.format(args.band))

min_loss = 1e8

def train(epoch):
    model.train()
    current_it = epoch * len(train_loader.dataset)

    modelpath = os.path.join(savepath, 'model_{}_{}.pt'.format(args.band, epoch))
    torch.save(model.state_dict(), modelpath)

    for batch_idx, (x, y, r) in enumerate(train_loader):
        current_it += min(args.batch_size, len(train_loader.dataset))

        if args.cuda:
            x = x.cuda().float()
            y = y.cuda().float()
            r = r.cuda().float()

        x = Variable(x, requires_grad=False)
        y = Variable(y, requires_grad=False)
        r = Variable(r, requires_grad=False)

        optimizer.zero_grad()
        pred = model(x, y)
        #loss = F.l1_loss(pred, y)
        loss = F.mse_loss(pred, r)

        loss.backward()
        optimizer.step()

        #for m in model.modules():
        #    if not hasattr(m, 'weight'):
        #        continue
        #    m.weight.data[::2, ::2] = (m.weight.data[::2, ::2] + m.weight.data[1::2, 1::2])/2
        #    m.weight.data[1::2, 1::2] = m.weight.data[::2, ::2]
        #    m.weight.data[1::2, ::2] = (m.weight.data[1::2, ::2] - m.weight.data[::2, 1::2])/2
        #    m.weight.data[::2, 1::2] = -m.weight.data[1::2, ::2]

        with open(trainlosspath, 'a') as f:
            f.write('{}\t{}\n'.format(current_it/len(train_loader.dataset), loss.data))

        if batch_idx % args.log_interval == 0:
            stdout = '[{}] Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f} {}'.format(            
                sec2hms(time.time()-st), epoch, batch_idx * args.batch_size, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data, args.postfix)
            print(stdout)

        wandb.log({'epoch': epoch, 'loss': loss.data, 'lr': optimizer.param_groups[0]['lr']})

def valid(epoch):
    global min_loss
    model.eval()
    valid_loss = 0
    for valid_it, (x, y, r) in enumerate(tqdm(valid_loader)):

        if args.cuda:
            x = x.cuda().float()
            y = y.cuda().float()
            r = r.cuda().float()

        x = Variable(x, requires_grad=False)
        y = Variable(y, requires_grad=False)
        r = Variable(r, requires_grad=False)

        pred = model(x, y)
        loss = F.l1_loss(pred[:,0], r[:,0])

        valid_loss += float(loss.mean().data.cpu())
         
    valid_loss /= len(valid_loader)
    with open(validlosspath, 'a') as f:
        f.write('{}\t{}\n'.format(epoch, valid_loss))
    if min_loss > loss.data:
        torch.save(model.state_dict(), bestmodelpath)
        print('best saved')
        min_loss = loss.data
    print('[{}] Valid Loss : {:.6f}'.format(sec2hms(time.time()-st), valid_loss))
            
def test(epoch):
    model.eval()
    test_loss = 0
    for test_it, (x, y, r) in enumerate(tqdm(test_loader)):
        if args.cuda:
            x = x.cuda().float()
            y = y.cuda().float()
            r = r.cuda().float()

        x = Variable(x, requires_grad=False)
        y = Variable(y, requires_grad=False)
        r = Variable(r, requires_grad=False)

        pred = model(x, y)
        loss = F.l1_loss(pred, r)

        test_loss += float(loss.mean().data.cpu())
    test_loss /= len(test_loader)
    with open(testlosspath, 'a') as f:
        f.write('{}\t{}\n'.format(epoch, test_loss))
    print('[{}] test Loss : {:.6f}'.format(sec2hms(time.time()-st), test_loss))

if __name__=='__main__':
    model = Net()
    if args.cuda:
        model.cuda()

    wandb.hook_torch(model, log='all')
    model = nn.DataParallel(model)

    print('[{:5.3f}] Model Created'.format(time.time()-st))

    train_loader = torch.utils.data.DataLoader(
        SimDataset(which_set='train', cache=args.cache, norm=args.norm, demix=args.demix), collate_fn=collate_fn, 
        batch_size=args.batch_size, shuffle=args.shuffle, **loader_kwargs)
    #valid_loader = torch.utils.data.DataLoader(
    #    SimDataset(which_set='dev', subset=args.subset, cache=args.cache), collate_fn=collate_fn, 
    #    batch_size=args.valid_batch_size, shuffle=args.shuffle, **loader_kwargs)
    #test_loader = torch.utils.data.DataLoader(
    #    SimDataset(which_set='test', subset=args.subset, cache=args.cache), collate_fn=collate_fn, 
    #    batch_size=args.valid_batch_size, shuffle=args.shuffle, **loader_kwargs)

    os.system('cp model.py dataset.py train.py {}'.format(savepath))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print('[{:5.3f}] Start Training'.format(time.time()-st))
    valid_interval_epochs = 15
    for epoch in range(args.epochs): 
        #if epoch % valid_interval_epochs ==0:
        #    valid(epoch)
        #test(epoch)
        train(epoch)
