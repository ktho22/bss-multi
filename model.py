import torch, torchvision
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import numpy as np


class Net(nn.Module):
    def __init__(self, including_dc=False, bandwidth=1, nsource=4, nch=6, nhid=400):
        nframes = 20

        if including_dc:
            bandwidth += 1

        super().__init__()

        self.act = nn.Tanh
        #self.model = nn.Sequential(
        #        #nn.Linear(4*(nsource*nch), nhid), 
        #        nn.Linear(2*(nsource+nch)*bandwidth + 4*(nsource+nch)**2, nhid), 
        #        #nn.Linear(6*2*(nsource+nch)*bandwidth + 36*4*(nsource*nch), nhid), 
        #        #nn.Linear(2*(nsource+nch)*bandwidth, nhid), 
        #        self.act(),
        #        nn.Linear(nhid, nhid), # in-channel: nhidden, out-channel: (6*4) * 2
        #        self.act(),
        #        nn.Linear(nhid, nhid), # in-channel: nhidden, out-channel: (6*4) * 2
        #        self.act(),
        #        nn.Linear(nhid, nhid), # in-channel: nhidden, out-channel: (6*4) * 2
        #        self.act(),
        #        nn.Linear(nhid, nhid), # in-channel: nhidden, out-channel: (6*4) * 2
        #        self.act(),
        #        nn.Linear(nhid, nhid), # in-channel: nhidden, out-channel: (6*4) * 2
        #        self.act(),
        #        nn.Linear(nhid, nhid), # in-channel: nhidden, out-channel: (6*4) * 2
        #        self.act(),
        #        nn.Linear(nhid, 2*nsource*bandwidth) # in-channel: nhidden, out-channel: (6*4) * 2
        #        #nn.Linear(nhid, 2*nch*nsource*bandwidth) # in-channel: nhidden, out-channel: (6*4) * 2
        #        )

        self.Wd = nn.Linear(2*(nsource)*bandwidth, nhid)
        #self.Wi = nn.Linear(2*(nch)*bandwidth, nhid)
        self.Wi = nn.Sequential(
                nn.Linear(2*(nch)*bandwidth, 12),
                self.act(),
                nn.Linear(12, 2),
                self.act(),
                nn.Linear(2, 12),
                self.act(),
                nn.Linear(12, nhid),
                )
        self.tanh = nn.Tanh()

        self.model = nn.Sequential(
                #nn.Linear(2*(nsource+nch)*bandwidth, nhid), 
                #nn.Linear(2*(nsource+nch)*bandwidth + 4*(nsource+nch)**2, nhid), 
                #self.act(),
                nn.Linear(nhid, nhid), # in-channel: nhidden, out-channel: (6*4) * 2
                self.act(),
                nn.Linear(nhid, 2*nsource*bandwidth) # in-channel: nhidden, out-channel: (6*4) * 2
                )
        self.init_weights()

    def forward(self, x, y):
        ## XY
        #xy = torch.bmm(x.unsqueeze(2),y.unsqueeze(1))
        #xy = xy.view(xy.shape[0], -1)
        #i = torch.cat((x, y, xy), dim=1)
        ##o = self.model(xy)
        #o = self.model(i)
        h1 = self.tanh(self.Wd(y) + self.Wi(x))
        o = self.model(h1)

        #i = torch.cat((x, y), dim=1)
        #o = self.model(i)

        #xy_cat = torch.cat((x, y), dim=1)
        #xy = torch.bmm(xy_cat.unsqueeze(2),xy_cat.unsqueeze(1))
        #xy = xy.view(xy.shape[0], -1)
        #i = torch.cat((x, y, xy), dim=1)
        #o = self.model(i)
        return o 

    def actmag(self, r, i):
        mag = r**2 + i **2
        pass

    def init_weights(self):
        for m in self.modules():
            if hasattr(m ,'weight'):
                nn.init.xavier_normal_(m.weight.data)
                #m.weight.data[::2, ::2] = (m.weight.data[::2, ::2] + m.weight.data[1::2, 1::2])/2
                #m.weight.data[1::2, 1::2] = m.weight.data[::2, ::2]
                #m.weight.data[1::2, ::2] = (m.weight.data[1::2, ::2] - m.weight.data[::2, 1::2])/2
                #m.weight.data[::2, 1::2] = -m.weight.data[1::2, ::2]
        Wu1 = self.model[0].weight
        Wu2 = self.model[2].weight
        self.Wd.weight.data = (Wu2 @ Wu1).permute(1,0).contiguous()

