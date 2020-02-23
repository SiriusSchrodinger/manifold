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
        # output = ((total 32 - kern) / stride) + 1
        self.spd_conv1 = spd.SPDConv2D(1, 4, 3, 1)
        self.spd_conv2 = spd.SPDConv2D(4, 8, 3, 1)
        self.spd_conv3 = spd.SPDConv2D(8, 16, 3, 1)
        self.spd_conv4 = spd.SPDConv2D(16, 16, 2, 1)
        self.spd_conv5 = spd.SPDConv2D(16, 8, 2, 1)

        
        self.spd_conv6 = spd.SPDConv2D(8, 16, 2, 1)
        self.spd_conv7 = spd.SPDConv2D(16, 16, 2, 1)
        self.spd_conv8 = spd.SPDConv2D(16, 8, 3, 1)
        self.spd_conv9 = spd.SPDConv2D(8, 4, 3, 1)
        self.spd_conv10 = spd.SPDConv2D(4, 1, 3, 1)

        
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

        x = padding(x, 2)
        print("six after padding")
        print(x.shape)
        x, wp4 = self.spd_conv6(x)
        x = padding(x, 2)
        x, wp4 = self.spd_conv7(x)
        x = padding(x, 4)
        x, wp4 = self.spd_conv8(x)
        x = padding(x, 4)
        x, wp5 = self.spd_conv9(x)
        x = padding(x, 4)
        x, wp6 = self.spd_conv10(x)

        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def padding(x, padding_dim):
    result = np.zeros((x.shape[0], x.shape[1], x.shape[2] + padding_dim, x.shape[3] + padding_dim, 3, 3))
    identity = np.identity(3)
    for m in range(result.shape[0]):
        for p in range(result.shape[1]):
            for i in range(result.shape[2]):
                for j in range(result.shape[3]):
                    if i < padding_dim or (result.shape[2] - i) <= padding_dim:
                        result[m][p][i][j] = identity
                    elif j < padding_dim or (result.shape[3] - j) <= padding_dim:
                        result[m][p][i][j] = identity
                    else:
                        result[m][p][i][j] = x[m][p][i - padding_dim][j - padding_dim]
    result = torch.from_numpy(result).to(device)
    return result


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
        sample_processed = sample_processed.reshape(1,64,64,3,3)[:,16:48,16:48,...]

        label = None

        #356
        if index < 356:
            label = 1
        else:
            label = 0

        return sample_processed, label

