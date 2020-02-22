import torch
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import spd

from batch_svd import batch_svd

class ManifoldNetSPD(nn.Module):
    def __init__(self):
        super(ManifoldNetSPD, self).__init__()
        #in_channel, out_channel, kernel_size, stride
        self.spd_conv1 = spd.SPDConv2D(1, 4, 5, 2)
        # output = ((total 64 - kern) / stride) + 1
        self.spd_conv2 = spd.SPDConv2D(4, 8, 5, 2)
        self.spd_conv3 = spd.SPDConv2D(8, 16, 3, 1)
        self.spd_conv4 = spd.SPDConv2D(16, 16, 2, 1)
        self.spd_conv5 = spd.SPDConv2D(16, 8, 2, 1)

        
    def forward(self, x):
        print("zero")
        print(x.shape)
        x,wp1 = self.spd_conv1(x)
        print("first")
        print(x.shape)
        x,wp2 = self.spd_conv2(x)
        print("second")
        print(x.shape)
        x,wp3 = self.spd_conv3(x)
        print("third")
        print(x.shape)
        x,wp3 = self.spd_conv4(x)
        print("forth")
        print(x.shape)
        x,wp3 = self.spd_conv5(x)
        print("fifth")
        print(x.shape)

        return x

class ParkinsonsDataset(data.Dataset):
  def __init__(self, data_tensorf):
        'Initialization'
        self.all_data = np.load(data_tensorf)['arr_0']
        self.all_data = torch.from_numpy(self.all_data)

  def __len__(self):
        'Denotes the total number of samples'
        return self.all_data.shape[0]

  def __getitem__(self, index):
        sample_processed = self.all_data[index].float()
        sample_processed = sample_processed.view(-1,3,3)
        
        samples = []
        for i in range(sample_processed.shape[0]):
            s,v = torch.symeig(sample_processed[i],eigenvectors=True)
            s = torch.diag(torch.clamp(s, min=0.0001))
            samples.append(torch.matmul(v, torch.matmul(s, v.t())))

        sample_processed = torch.stack(samples)
        sample_processed = sample_processed.reshape(1,64,64,3,3)[:,16:47,16:47,...]

        label = None

        #356
        if index < 356:
            label = 1
        else:
            label = 0

        return sample_processed, label

