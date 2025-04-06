import argparse
import random
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns

class GCNModel(nn.Module):
    def __init__(self, args, seq_feature=None, structure_feature=None, feature_fusion=None):
        super(GCNModel, self).__init__()

        self.seq_dim = seq_feature.shape[-1]
        self.structure_dim = structure_feature.shape[-1]
        self.fusion_type = feature_fusion
        self.structure_feature = structure_feature
        self.seq_feature = seq_feature

        # fusion type
        if self.fusion_type == 'concat':
            self.layer1_f = nn.Sequential(nn.Linear(self.structure_dim + self.seq_dim, self.seq_dim),
                                          nn.BatchNorm1d(self.seq_dim),
                                          nn.LeakyReLU(True))
            self.layer2_f = nn.Sequential(nn.Linear(self.seq_dim, self.seq_dim), nn.BatchNorm1d(self.seq_dim),
                                          nn.LeakyReLU(True))
            self.layer3_f = nn.Sequential(nn.Linear(self.seq_dim, self.seq_dim), nn.BatchNorm1d(self.seq_dim),
                                          nn.LeakyReLU(True))

        elif self.fusion_type == 'sum':
            self.W_s = nn.Linear(self.structure_dim, self.seq_dim)
            self.W_e = nn.Linear(self.seq_dim, self.seq_dim)

        elif self.fusion_type == 'cross_fusion':
            self.protein_structure = nn.Linear(self.structure_dim, self.seq_dim)
            self.protein_seq = nn.Linear(self.seq_dim, self.seq_dim)

            self.add_drug = nn.Sequential(nn.Linear(self.seq_dim, self.seq_dim))
            self.cross_add_drug = nn.Sequential(nn.Linear(self.seq_dim, self.seq_dim))
            self.multi_drug = nn.Sequential(nn.Linear(self.seq_dim, self.seq_dim))
            self.activate = nn.ReLU()

            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(5, 5)),
                nn.BatchNorm2d(8), nn.MaxPool2d((2, 2)), nn.ReLU())
            self.conv2 = nn.Sequential(
                nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(5, 5)),
                nn.BatchNorm2d(8), nn.MaxPool2d((2, 2)), nn.ReLU())
            if self.seq_dim == 768:
                self.fc1 = nn.Sequential(nn.Linear(189 * 189 * 8, self.seq_dim), nn.BatchNorm1d(self.seq_dim),
                                         nn.ReLU(True))
            elif self.seq_dim == 512:
                self.fc1 = nn.Sequential(nn.Linear(125 * 125 * 8, self.seq_dim), nn.BatchNorm1d(self.seq_dim),
                                         nn.ReLU(True))
            elif self.seq_dim == 300:
                self.fc1 = nn.Sequential(nn.Linear(72 * 72 * 8, self.seq_dim), nn.BatchNorm1d(self.seq_dim),
                                         nn.ReLU(True))
            elif self.seq_dim == 256:
                self.fc1 = nn.Sequential(nn.Linear(61 * 61 * 8, self.seq_dim), nn.BatchNorm1d(self.seq_dim),
                                         nn.ReLU(True))
            elif self.seq_dim == 128:
                self.fc1 = nn.Sequential(nn.Linear(29 * 29 * 8, self.seq_dim), nn.BatchNorm1d(self.seq_dim),
                                         nn.ReLU(True))
            elif self.seq_dim == 100:
                self.fc1 = nn.Sequential(nn.Linear(22 * 22 * 8, self.seq_dim), nn.BatchNorm1d(self.seq_dim),
                                         nn.ReLU(True))
            elif self.seq_dim == 32:
                self.fc1 = nn.Sequential(nn.Linear(5 * 5 * 8, self.seq_dim), nn.BatchNorm1d(self.seq_dim),
                                         nn.ReLU(True))
            elif self.seq_dim == 64:
                self.fc1 = nn.Sequential(nn.Linear(13 * 13 * 8, self.seq_dim), nn.BatchNorm1d(self.seq_dim),
                                         nn.ReLU(True))

            self.fc2_global = nn.Sequential(
                nn.Linear(self.seq_dim * self.seq_dim + self.seq_dim, self.seq_dim),
                nn.ReLU(True))
            self.fc2_global_reverse = nn.Sequential(
                nn.Linear(self.seq_dim * self.seq_dim + self.seq_dim, self.seq_dim),
                nn.ReLU(True))
            self.fc2_cross = nn.Sequential(
                nn.Linear(self.seq_dim * 5, self.seq_dim),
                nn.ReLU(True))

    def generate_fusion_feature(self, seq_batch, structure_batch, batch_data):

        if self.fusion_type == 'concat':
            x = torch.cat([seq_batch, structure_batch], dim=1)
            x = self.layer1_f(x)
            x = self.layer2_f(x)
            x = self.layer3_f(x)
            return x

        elif self.fusion_type == 'sum':
            structure = self.W_s(self.structure_feature)
            seq = self.W_e(self.seq_feature)
            add_structure_seq = structure + seq
            return add_structure_seq

        elif self.fusion_type == 'cross_fusion':
            seq = self.protein_seq(structure_batch)
            structure = self.protein_structure(seq_batch)
            seq_embed_reshape = seq.unsqueeze(-1)
            structure_embed_reshape = structure.unsqueeze(-1)
            seq_matrix = seq_embed_reshape * structure_embed_reshape.permute((0, 2, 1))
            seq_matrix_reverse = structure_embed_reshape * seq_embed_reshape.permute((0, 2, 1))
            seq_global = seq_matrix.view(seq_matrix.size(0), -1)
            seq_global_reverse = seq_matrix_reverse.view(seq_matrix_reverse.size(0), -1)
            seq_matrix_reshape = seq_matrix.unsqueeze(1)
            seq_matrix_reshape = seq_matrix_reshape.to('cuda')
            seq_data = seq_matrix_reshape
            out = self.conv1(seq_data)
            out = self.conv2(out)
            out = out.view(out.size(0), -1)
            embedding_data = self.fc1(out)
            global_local_before = torch.cat((embedding_data, seq_global), 1)
            cross_embedding_pre = self.fc2_global(global_local_before)

            # another reverse part
            seq_matrix_reshape_reverse = seq_matrix_reverse.unsqueeze(1)
            seq_matrix_reshape_reverse = seq_matrix_reshape_reverse.to('cuda')
            seq_reverse = seq_matrix_reshape_reverse
            out = self.conv1(seq_reverse)
            out = self.conv2(out)
            out = out.view(out.size(0), -1)
            embedding_data_reverse = self.fc1(out)
            global_local_before_reverse = torch.cat((embedding_data_reverse, seq_global_reverse), 1)
            cross_embedding_pre_reverse = self.fc2_global_reverse(global_local_before_reverse)
            out3 = self.activate(self.multi_drug(structure * seq))
            out_concat = torch.cat((structure_batch, seq_batch, cross_embedding_pre, cross_embedding_pre_reverse, out3), 1)
            out_concat = self.fc2_cross(out_concat)
            return out_concat

    def forward(self, *input):
        return self.generate_fusion_feature(*input)
    
def cross_fusion(args):
    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    
    with open(args.seq_feature_path, "r") as seq_json:
        seq = np.array(json.load(seq_json))
    seq = seq.reshape(seq.shape[0], 768)
    seq = torch.tensor(seq, dtype=torch.float32).to(device)
    
    with open(args.structure_feature_path, "r") as structure_json:
        structure = np.array(json.load(structure_json))
    structure = seq.reshape(structure.shape[0], 768)
    structure = torch.tensor(structure, dtype=torch.float32).to(device)
    
    indices = torch.arange(seq.size(0))
    dataset = TensorDataset(seq, structure, indices)
    
    model = GCNModel(args=args, seq_feature=seq, structure_feature=structure, feature_fusion=args.feature_fusion)
    model.to(device)
    
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    results = []
    indices_list = []
    batch_results = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Processing batches", unit="batch"):
            seq_batch, structure_batch, batch_indices = batch
            seq_batch = seq_batch.to(device)
            structure_batch = structure_batch.to(device)
            batch_indices = batch_indices.to(device)

            fused_feature = model(seq_batch, structure_batch, batch_indices)
            results.extend(fused_feature.cpu().tolist())
    
    with open(args.encoded_proteins_cross_fusion_path, 'w') as f:
        json.dump(results, f)
    print("Fused Features Shape:", len(results))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run MMF2Drug.")

    parser.add_argument('--seed', type=int, default=2024, help='Random seed.')
    parser.add_argument('--seq_feature_path', type=str, default='./storage/cross_dock/encoded_proteins_seq_train.latent', help='sequence feature path')
    parser.add_argument('--structure_feature_path', type=str, default='./storage/cross_dock/encoded_proteins_hypergraph_train.latent', help='structure feature path')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--feature_fusion', nargs='?', default='concat', help='特征融合类型：concat / sum / cross_fusion')
    parser.add_argument('--encoded_proteins_cross_fusion_path', type=str, default='./storage/cross_dock/encoded_proteins_concat_fusion_train.latent', help='result save path')
    args = parser.parse_args()
    cross_fusion(args)