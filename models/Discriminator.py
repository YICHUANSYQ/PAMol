import torch
import torch.nn as nn
import numpy as np


class Discriminator(nn.Module):
    def __init__(self, data_shape):
        super(Discriminator, self).__init__()
        self.data_shape = data_shape

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.data_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, mol):
        validity = self.model(mol)
        return validity

    def save(self, path):
        save_dict = {
            'model': self.model.state_dict(),
            'data_shape': self.data_shape,
        }
        torch.save(save_dict, path)
        return

    @staticmethod
    def load(path):
        save_dict = torch.load(path)
        D = Discriminator(save_dict['data_shape'])
        D.model.load_state_dict(save_dict["model"])

        return D


class C_Discriminator(nn.Module):
    def __init__(self, data_shape):
        super(C_Discriminator, self).__init__()
        self.data_shape = data_shape
        self.condition_dim = 768

        self.condition_MLE = nn.Linear(768, 512)

        self.seq_dim = 512
        self.structure_dim = 512
        
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
            nn.Linear(self.seq_dim * 5, self.seq_dim * 2),
            nn.ReLU(True))

        self.model = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, mol, condition):
        condition = condition.to(torch.float32)
        condition = self.condition_MLE(condition)
        feature = torch.cat((mol, condition), dim=1)

        # seq_embed_reshape = mol.unsqueeze(-1)  # 扩展最后一维，形状：batch_size * embed_dim * 1
        # structure_embed_reshape = condition.unsqueeze(-1)  # 扩展最后一维，形状：batch_size * embed_dim * 1
        # seq_matrix = seq_embed_reshape * structure_embed_reshape.permute((0, 2, 1))  # 计算交叉矩阵 形状：batch_size * embed_dim * embed_dim
        # seq_matrix_reverse = structure_embed_reshape * seq_embed_reshape.permute((0, 2, 1))
        # seq_global = seq_matrix.view(seq_matrix.size(0), -1)
        # seq_global_reverse = seq_matrix_reverse.view(seq_matrix_reverse.size(0), -1)

        # seq_matrix_reshape = seq_matrix.unsqueeze(1) # 在第一维新增一个维度，用于后续卷积操作，形状变为 batch_size * 1 * embed_dim * embed_dim
        # seq_matrix_reshape = seq_matrix_reshape.to('cuda')
        # seq_data = seq_matrix_reshape
        # out = self.conv1(seq_data)
        # out = self.conv2(out)
        # # 将卷积后的特征展平
        # out = out.view(out.size(0), -1)
        # # 全连接层
        # embedding_data = self.fc1(out)
        # # 将局部特征 (embedding_data) 与全局特征 (seq_global) 拼接
        # global_local_before = torch.cat((embedding_data, seq_global), 1)
        # # 通过全连接层融合局部和全局特征，得到正向融合特征 cross_embedding_pre
        # cross_embedding_pre = self.fc2_global(global_local_before)

        # # another reverse part
        # seq_matrix_reshape_reverse = seq_matrix_reverse.unsqueeze(1)
        # # 局部特征提取 逆向操作
        # seq_matrix_reshape_reverse = seq_matrix_reshape_reverse.to('cuda')
        # seq_reverse = seq_matrix_reshape_reverse
        # # 对特征进行卷积，提取局部特征
        # out = self.conv1(seq_reverse)
        # out = self.conv2(out)
        # # 将卷积后的特征展平
        # out = out.view(out.size(0), -1)
        # # 全连接层
        # embedding_data_reverse = self.fc1(out)
        # # 拼接逆向局部特征与全局特征 (seq_global_reverse)
        # global_local_before_reverse = torch.cat((embedding_data_reverse, seq_global_reverse), 1)
        # # 通过全连接层融合局部和全局特征，得到逆向融合特征 cross_embedding_pre_reverse
        # cross_embedding_pre_reverse = self.fc2_global_reverse(global_local_before_reverse)
        # out3 = self.activate(self.multi_drug(mol * condition))
        # out_concat = torch.cat((mol, condition, cross_embedding_pre, cross_embedding_pre_reverse, out3), 1)
        # out_concat = self.fc2_cross(out_concat)
        
        # validity = self.model(out_concat)
        validity = self.model(feature)
        return validity

    def save(self, path):
        save_dict = {
            'model': self.model.state_dict(),
            'data_shape': self.data_shape,
        }
        torch.save(save_dict, path)
        return

    @staticmethod
    def load(path):
        save_dict = torch.load(path)
        D = C_Discriminator(save_dict['data_shape'])
        D.model.load_state_dict(save_dict["model"])

        return D
