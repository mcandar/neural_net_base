import numpy as np

def split(x,ratios):
    size = x.shape[0]
    ratios = np.r_[0,np.cumsum(ratios)]
    idx = (size*ratios).astype(int)
    return [x[i:j] for i,j in zip(idx[:-1],idx[1:])]

def prep_batch(batch_size,*args):
    N = len(args[0])
    seq = np.r_[np.arange(0,N,batch_size),N-1]
    for i,j in np.c_[seq[:-1],seq[1:]]:
        yield (item[i:j] for item in args)