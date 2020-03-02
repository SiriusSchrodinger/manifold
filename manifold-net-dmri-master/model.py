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
        first_start = time.time()
        self.spd_conv1 = spd.SPDConv2D(1, 4, 3, 1)
        first_end = time.time()
        print(first: first_end - first_start)
        second_start = time.time()
        self.spd_conv2 = spd.SPDConv2D(4, 8, 3, 1)
        second_end = time.time()
        print(second: second_end - second_start)
        third_start = time.time()
        self.spd_conv3 = spd.SPDConv2D(8, 16, 3, 1)
        third_end = time.time()
        print(third: third_end - third_start)
        #self.spd_conv4 = spd.SPDConv2D(16, 16, 3, 1)
        #self.spd_conv5 = spd.SPDConv2D(16, 8, 3, 1)


        #self.spd_conv6 = spd.CayleyConv(8, 16, 3, 1)
        #self.spd_conv7 = spd.CayleyConv(16, 16, 3, 1)
        forth_start = time.time()
        self.spd_conv8 = spd.CayleyConv(16, 8, 3, 1)
        forth_end = time.time()
        print(first: forth_end - forth_start)
        fifth_start = time.time()
        self.spd_conv9 = spd.CayleyConv(8, 4, 3, 1)
        fifth_end = time.time()
        print(fifth: fifth_end - fifth_start)
        six_start = time.time()
        self.spd_conv10 = spd.CayleyConv(4, 1, 3, 1)
        six_end = time.time()
        print(fifth: six_end - six_start)


    def forward(self, x):
        #import pdb; pdb.set_trace()
        x,wp1 = self.spd_conv1(x)
        x,wp2 = self.spd_conv2(x)
        x,wp3 = self.spd_conv3(x)
        #x,wp4 = self.spd_conv4(x)
        #x,wp5 = self.spd_conv5(x)

        #x, wp6 = self.spd_conv6(x)
        #x, wp7 = self.spd_conv7(x)
        x, wp8 = self.spd_conv8(x)
        x, wp9 = self.spd_conv9(x)
        x, wp10 = self.spd_conv10(x)

        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def padding(x, padding_dim):
    result = torch.ones([x.shape[0], x.shape[1], x.shape[2] + padding_dim * 2, x.shape[3] + padding_dim * 2, 3, 3], device = device)
    for m in range(result.shape[0]):
        for p in range(result.shape[1]):
            for i in range(result.shape[2]):
                for j in range(result.shape[3]):
                    if i < padding_dim or (result.shape[2] - i) <= padding_dim:
                        result[m][p][i][j] = torch.eye(3, device = device)
                    elif j < padding_dim or (result.shape[3] - j) <= padding_dim:
                        result[m][p][i][j] = torch.eye(3, device = device)
                    else:
                        result[m][p][i][j] = x[m][p][i - padding_dim][j - padding_dim]
    return result

def add_identity(x):
    for m in range(x.shape[0]):
        for p in range(x.shape[1]):
            for i in range(x.shape[2]):
                for j in range(x.shape[3]):
                    x[m][p][i][j] = torch.add(x[m][p][i][j], torch.eye(3, device = device))


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
        sample_processed = sample_processed.reshape(1,64,64,3,3)[0:1712,16:48,16:48,...]

        return sample_processed

