from torch import nn
from hgnn_model import HGNN_conv, HGNN_fc
from torch_geometric.nn import Set2Set
import torch.nn.functional as F

class HGNN(nn.Module):
    def __init__(self, in_ch, out_size, n_hid, dropout=0.5):
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, out_size)
        
        self.set2set = Set2Set(in_channels=out_size, processing_steps=5, num_layers=2)

    def forward(self, x, G):
        x = F.relu(self.hgc1(x, G))
        x = F.dropout(x, self.dropout)
        x = F.relu(self.hgc2(x, G))
        return self.set2set(x)
