import torch 
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import fm_ops as spd_ops
from batch_svd import batch_svd


def weightNormalize(weights):
    out = []
    for row in weights.view(weights.shape[0],-1):
         out.append(torch.clamp(row, min=0.001, max=0.999))
    return torch.stack(out).view(*weights.shape)

class CayleyConv(nn.Module):
    def __init__(self, in_channels, out_channels, kern_size, stride):
        super(CayleyConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kern_size = kern_size
        self.stride = stride
        self.g = torch.nn.Parameter(self.create_weight(),requires_grad=True)

    def create_weight(self):
        x = (torch.rand([self.out_channels, self.in_channels, 8, 3]) - 0.5) * 0.1
        return x

    def inverse3(self, b_mat):
        eps = 0.0000001
        b00 = b_mat[:,0,0]
        b01 = b_mat[:,0,1]
        b02 = b_mat[:,0,2]
        b10 = b_mat[:,1,0]
        b11 = b_mat[:,1,1]
        b12 = b_mat[:,1,2]
        b20 = b_mat[:,2,0]
        b21 = b_mat[:,2,1]
        b22 = b_mat[:,2,2]
        det = (b00*(b11*b22-b12*b21)-b01*(b10*b22-b12*b20)+b02*(b10*b21-b11*b20))
        c00 = b11*b22 - b12*b21
        c01 = b02*b21 - b01*b22
        c02 = b01*b12 - b02*b11
        c10 = b12*b20 - b10*b22
        c11 = b00*b22 - b02*b20
        c12 = b02*b10 - b00*b12
        c20 = b10*b21 - b11*b20
        c21 = b01*b20 - b00*b21
        c22 = b00*b11 - b01*b10
        c00 = (c00/ (det+eps)).view(-1, 1, 1)
        c01 = (c01/ (det+eps)).view(-1, 1, 1)
        c02 = (c02/ (det+eps)).view(-1, 1, 1)
        c10 = (c10/ (det+eps)).view(-1, 1, 1)
        c11 = (c11/ (det+eps)).view(-1, 1, 1)
        c12 = (c12/ (det+eps)).view(-1, 1, 1)
        c20 = (c20/ (det+eps)).view(-1, 1, 1)
        c21 = (c21/ (det+eps)).view(-1, 1, 1)
        c22 = (c22/ (det+eps)).view(-1, 1, 1)
        b_inv1 = torch.cat((torch.cat((c00,c01,c02), dim=2), torch.cat((c10,c11,c12), dim=2), torch.cat((c20,c21,c22), dim=2)), dim=1)
        return b_inv1        

    def forward(self, x):
        # x = [batch, in, row, col, 3, 3]
        # g = [out, in, 8, 3]
        #assume stride = 1
        #assume ker = 3
        result = torch.zeros([x.shape[0], self.out_channels, x.shape[2] - 1 + self.kern_size, x.shape[3] - 1 + self.kern_size, 3, 3]).cuda()
        x_matrix = torch.zeros([self.out_channels, self.in_channels, 3, 3, 3, 3]).cuda()
        identitys = torch.zeros([self.out_channels, self.in_channels, 3, 3, 3, 3])
        for i in range(x_matrix.shape[0]):
            for j in range(x_matrix.shape[1]):
                for k in range(x_matrix.shape[2]):
                    for l in range(x_matrix.shape[3]):
                        if k == 1 and l == 1:
                            x_matrix[i][j][k][l] = torch.eye(3)
                            identitys[i][j][k][l] = torch.eye(3)
                        else:
                            num = k * 3 + l
                            if num == 8:
                                num = 4
                            _a = self.g[i][j][num][0]
                            _b = self.g[i][j][num][1]
                            _c = self.g[i][j][num][2]
                            x_matrix[i][j][k][l][0][0] = 0
                            x_matrix[i][j][k][l][0][1] = _a
                            x_matrix[i][j][k][l][0][2] = _b
                            x_matrix[i][j][k][l][1][0] = -_a
                            x_matrix[i][j][k][l][1][1] = 0
                            x_matrix[i][j][k][l][1][2] = _c
                            x_matrix[i][j][k][l][2][0] = -_b
                            x_matrix[i][j][k][l][2][1] = -_c
                            x_matrix[i][j][k][l][2][2] = 0

                            identitys[i][j][k][l][0][0] = 1
                            identitys[i][j][k][l][0][1] = -_a
                            identitys[i][j][k][l][0][2] = -_b
                            identitys[i][j][k][l][1][0] = _a
                            identitys[i][j][k][l][1][1] = 1
                            identitys[i][j][k][l][1][2] = -_c
                            identitys[i][j][k][l][2][0] = _b
                            identitys[i][j][k][l][2][1] = _c
                            identitys[i][j][k][l][2][2] = 1

        inverse_prep = self.inverse3(identitys.view([self.out_channels * self.in_channels * 3 * 3, 3, 3]))
        inversed = inverse_prep.view(self.out_channels, self.in_channels, 3, 3, 3, 3).cuda()
        for m in range(result.shape[0]):
            for o in range(self.out_channels):
                for i in range(self.in_channels):
                    for r in range(x.shape[2]):
                        for c in range(x.shape[3]):
                            #center = [r + 1][c + 1]
                            for a in range(3):
                                for b in range(3):
                                    g_matrix = torch.mm(inversed[o][i][a][b], (torch.eye(3).cuda() + x_matrix[o][i][a][b]))
                                    result[m][o][r + a][c + b] += torch.mm(torch.mm(g_matrix, x[m][i][r][c]), g_matrix.t())
                            result[m][o][r + 1][c + 1] = x[m][i][r][c]
        return result, 0


class SPDConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kern_size, stride):
        super(SPDConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kern_size = kern_size
        self.stride = stride
        self.weight_matrix = torch.nn.Parameter(torch.rand(out_channels, (kern_size**2)*in_channels),requires_grad=True)

    # x: [batches, channels, rows, cols, 3, 3]
    def forward(self, x):
        ####
        #out = x.view(-1,3,3)
        #for i in range(out.shape[0]):
        #    if torch.det(out[i]) <= 0:
        #        print(torch.eig(out[i]))
        ####

        # x: [batches, channels, rows, cols, 3, 3] -> 
        #    [batches, channels, 3, 3, rows, cols]
        x = x.permute(0,1,4,5,2,3).contiguous()

        # x_windows: [batches, channels, 3, 3, rows_reduced, cols_reduced, window_x, window_y]
        x_windows = x.unfold(4, self.kern_size, self.stride).contiguous()
        x_windows = x_windows.unfold(5, self.kern_size, self.stride).contiguous()

        x_s = x_windows.shape
        #x_windows: [batches, channels, 3, 3,  rows_reduced, cols_reduced, window]   
        x_windows = x_windows.view(x_s[0],x_s[1],x_s[2],x_s[3],x_s[4],x_s[5],-1)

        #x_windows: [batches, rows_reduced, cols_reduced, window, channels, 3,3]
        x_windows = x_windows.permute(0,4,5,6,1,2,3).contiguous()

        x_s = x_windows.shape
        x_windows = x_windows.view(x_s[0],x_s[1],x_s[2],-1,x_s[5],x_s[6]).contiguous()


        #Output format: [batches, sequence, out_channels, cov_x, cov_y]
        return spd_ops.recursiveFM2D(x_windows, weightNormalize(self.weight_matrix)), 0



class SPDLinear(nn.Module):
    def __init__(self):
        super(SPDLinear, self).__init__()
        self.A = torch.rand(2,288).cuda()

    #X: [-1, 3,3]
    #Y: [-1, 3,3]
    def GLmetric(self, X, Y):
        inner = torch.matmul(torch.inverse(X), Y)


        u,s,v = batch_svd(inner)
        s_log = torch.diag_embed(torch.log(s))
        log_term = torch.matmul(u,torch.matmul(s_log,v.permute(0,2,1)))
        dist = torch.sum(torch.diagonal(torch.matmul(log_term,log_term), dim1=-2, dim2=-1),1)
        return dist
    
    #x: [batch, channels, rows, cols, 3,3]
    def forward(self, x):
        x_s = x.shape

        #x: [batch*channels, rows*cols, 3,3]
        x = x.view(x.shape[0]*x.shape[1], -1, x.shape[4], x.shape[5])

        #x: [batch*channels, 1, 1, rows*cols, 3,3]
        x = x.unsqueeze(1).unsqueeze(2)


        #weights: [1,rows*cols-1]
        weights = (1.0/torch.arange(start=2.0,end=x.shape[3]+1)).unsqueeze(0).cuda()
        
        #unweightedFM: [batches*channels, 1,1,1, 3,3]
        unweighted_FM = spd_ops.recursiveFM2D(x,weights)


        #unweightedFM: [batches*channels,3,3]
        unweighted_FM = unweighted_FM.view(-1, x_s[4], x_s[5])
        
        #unweightedFM: [batches*channels,rows*cols,3,3]
        unweighted_FM = unweighted_FM.unsqueeze(1).repeat(1, x_s[2]*x_s[3], 1, 1)

        #unweightedFM: [batches*channels*rows*cols,3,3]
        unweighted_FM = unweighted_FM.view(-1, x_s[4], x_s[5])


        #x: [batches*channels,rows*cols,3,3]
        x = x.view(-1, x_s[2]*x_s[3], x_s[4], x_s[5])
        #x: [batches*channels*rows*cols,3,3]
        x = x.view(-1, x_s[4], x_s[5])

        out = self.GLmetric(x, unweighted_FM)

        #out: [batch, channels*rows*cols]
        out = out.view(x_s[0], x_s[1]*x_s[2]*x_s[3])


        return out

