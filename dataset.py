import os, re, random, sys, time
from os.path import join, splitext, basename
from glob import glob
import numpy as np
from numpy import linalg
random.seed(123)

import torch
import torch.utils.data as data

from scipy import io
from scipy.fftpack import fft
import soundfile as sf
import librosa, itertools

class BSSDataset(data.Dataset):
    def __init__(self, which_set='train'):
        self.__dict__.update(locals())
        self.datapath = join('/hdd1/home/thkim/data/Dereverb_4IR_no_noise', self.which_set, '*mag.pth')
        
        self.datalist = sorted(glob(self.datapath))
        
    def __getitem__(self, index):
        magname = self.datalist[index]
        phsname = magname.replace('mag', 'phs')
        
        mag = torch.load(magname)
        phs = torch.load(phsname)

        return phs, self.unwrap(phs)


    def __len__(self):
        return len(self.datalist)


    def unwrap(self, phs, discont=np.pi, start_freq=128):
        """
        Unwrapping phase from the given frequency

        Args:
            phs (torch.tensor) [nMic, Freq, time]: phase information
            discont (float): discontinuiuty
            start_freq (int): from this freq, unwrapping freq both sides
        Return:
            unwrapped_phs (torch.tensor) [nMic, Freq, time]: Unwrapped phase
        """

        phs = phs.numpy()

        rhs = phs[:, start_freq:, :]
        lhs = phs[:, :start_freq+1, :]
        lhs = lhs[:, ::-1, :]

        unwrapped_rhs = np.unwrap(rhs, axis=1)
        unwrapped_lhs = np.unwrap(lhs, axis=1)

        unwrapped_phs = np.zeros_like(phs)

        unwrapped_phs[:, start_freq:, :] = unwrapped_rhs
        unwrapped_phs[:, :start_freq, :] = unwrapped_lhs[:, start_freq:0:-1, :]

        return torch.from_numpy(unwrapped_phs)
            

if __name__=='__main__':
    aa = BSSDataset()
    aa[0]
