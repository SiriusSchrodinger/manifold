import torch 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from batch_svd import batch_svd

def weightnormalize(weights):
    out = []
    for row in weights:
         out.append(row**2/torch.sum(row**2))
    return torch.stack(out)


def mexp(x, exp):
    u,s,v = batch_svd(x)
    ep = torch.diag_embed(torch.pow(s,exp))
    v = torch.einsum('...ij->...ji', v)
    return torch.matmul(torch.matmul(u,ep),v)


#M: [-1,3,3] 
#N: [-1,3,3]
#w: [-1]
def batchGLMean(M,N,w):
    
    # w:[-1, 3, 3]
    #w = w.unsqueeze(1).repeat(1,M.shape[-1])

    #u,s,v = batch_svd(M)
    #s_pow = torch.diag_embed(torch.pow(s,0.5))
    
    #M_sqrt = torch.matmul(u, torch.matmul(s_pow, v.permute(0,2,1)))

    #M_sqrt_inv = b_inv33(M_sqrt)

    #inner_term = torch.matmul(M_sqrt_inv, torch.matmul(N, M_sqrt_inv))

    
    #u_i, s_i, v_i = batch_svd(inner_term)


    #s_i_c = s_i.view(-1)
    #s_i_c_pow = s_i_c**(w.view(-1))
    #s_i_pow = s_i_c_pow.view(*s.shape)

    #s_i_pow = torch.diag_embed(s_i_pow)

    #inner_term_weighted = torch.matmul(u_i, torch.matmul(s_i_pow, v_i.permute(0,2,1)))


    #return torch.matmul(M_sqrt, torch.matmul(inner_term_weighted, M_sqrt))
    w = w.view((-1, 1, 1))
    return M * w + N * (1 - w)


#windows: [batches, rows_reduced, cols_reduced, window, 3, 3]
#weights: [out_channels, in_channels*kern_size**2}
def recursiveFM2D(windows, weights):
    w_s = windows.shape

    # windows: [batches*rows_reduced*cols_reduced, window, 3, 3]
    windows = windows.view(-1, windows.shape[3], windows.shape[4], windows.shape[5])

    oc = weights.shape[0]

    # weights: [batches*rows_reduced*cols_reduced, out_channels, in_channels*kern_size**2]\
    weights = weights.unsqueeze(0).repeat(windows.shape[0],1,1)

    # weights: [batches*rows_reduced*cols_reduced*out_channels, in_channels*kern_size**2]\
    weights = weights.view(-1, weights.shape[2])

    # [batches*rows_reduced*cols_reduced*channels_out, 3,3]
    running_mean = windows[:,0,:,:].unsqueeze(1).repeat(1,oc,1,1)
    running_mean = running_mean.view(-1,running_mean.shape[2], running_mean.shape[3])


    for i in range(1,weights.shape[1]):
        current_fiber = windows[:,i,:,:]
        
        #[batches*rows_reduced*cols_reduced, channels_out, 3, 3]
        current_fiber = current_fiber.unsqueeze(1).repeat(1,oc,1,1)
        cf_s = current_fiber.shape
        
        # [batches*rows_reduced*cols_reduced*channels_out, 3, 3]
        current_fiber = current_fiber.view(-1, cf_s[2], cf_s[3])


        running_mean = batchGLMean(current_fiber, running_mean, weights[:,i])

    #out: [batches, rows_reduced, cols_reduced, channels_out, 3, 3]
    out = running_mean.view(w_s[0], w_s[1], w_s[2], oc, w_s[4], w_s[5])

    #out: [batches, channels_out, rows_reduced, cols_reduced, 3, 3]
    out = out.permute(0,3,1,2,4,5).contiguous()
    
    return out


def b_inv33(b_mat):

    #b_mat = b_mat.cpu()

    #eye = b_mat.new_ones(b_mat.size(-1)).diag().expand_as(b_mat)

    #b_inv, _ = torch.gesv(eye, b_mat)

    #b_inv = b_inv.to(device)

    #print(b_inv.contiguous())

    #b = [t.inverse() for t in torch.unbind(b_mat)]

    #b_inv = torch.stack(b)
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









