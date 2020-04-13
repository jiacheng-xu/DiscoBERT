f = '/scratch/cluster/jcxu/data_intern/cnndm-bert/test/cnn.test.0.bert.pt'
import torch
with open(f,'rb') as fd:
    data = torch.load(fd)
print('')