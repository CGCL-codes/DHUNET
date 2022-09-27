import random

from torch.nn import functional as F
from torch import nn
import torch
from torch.nn.parameter import Parameter
import math
import os
import numpy as np
path_dir = os.getcwd()

class ConvTransE(torch.nn.Module):
    def __init__(self, num_entities, embedding_dim, gpu, layer_norm, d_k=64, d_v=64, n_heads=8, d_ff=2048, n_layers_=1, input_dropout=0, hidden_dropout=0, feature_map_dropout=0, channels=50, kernel_size=3, use_bias=True):

        super(ConvTransE, self).__init__()

        self.inp_drop = torch.nn.Dropout(input_dropout)
        self.hidden_drop = torch.nn.Dropout(hidden_dropout)
        self.feature_map_drop = torch.nn.Dropout(feature_map_dropout)
        self.loss = torch.nn.BCELoss()

        self.conv1 = torch.nn.Conv1d(2, channels, kernel_size, stride=1,
                               padding=int(math.floor(kernel_size / 2)))  # kernel size is odd, then padding = math.floor(kernel_size/2)
        self.bn0 = torch.nn.BatchNorm1d(2)
        self.bn1 = torch.nn.BatchNorm1d(channels)
        self.bn2 = torch.nn.BatchNorm1d(embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(embedding_dim * channels, embedding_dim)
        self.bn3 = torch.nn.BatchNorm1d(embedding_dim)
        # self.bn4 = torch.nn.BatchNorm1d(Config.embedding_dim)
        self.bn_init = torch.nn.BatchNorm1d(embedding_dim)
        self.gpu = gpu
        self.layer_norm = layer_norm
        self.output = None
        self.weight1 = None
        self.weight2 = None
        self.tanh = nn.Tanh()
        self.layers = nn.ModuleList([DecoderLayer(embedding_dim, d_k, d_v, n_heads, d_ff, gpu) for _ in range(n_layers_)])

    def forward(self, embedding, emb_rel, triplets, history_tail_seq, one_hot_tail_seq, layer_norm, use_cuda, nodes_id=None, mode="train", negative_rate=0, partial_embeding=None):
        '''
        传入每个历史子图中的实体和关系的分布式embedding，并读取valid或者test集中记录每个事实的历史重复事实的本地文件
        :param embedding: (num_ents, h_dim) 在gpu上
        :param emb_rel: (num_rels*2, h_dim)
        :param triplets: 包含反关系的valid/test集的一个时间戳的triples二维张量 (num_triples(batch_size), 3)
        :param history_tail_seq:
        :param one_hot_tail_seq:
        :param nodes_id:
        :param mode:
        :param negative_rate:
        :param partial_embeding:
        :return:
        '''
        batch_size = len(triplets)
        if len(embedding) == 1:
            # Number 0
            evolve_emb_0 = embedding[0]
            # 实体的分布式嵌入
            e1_embedded_all = F.normalize(evolve_emb_0) if layer_norm else evolve_emb_0
            e1_embedded_all = F.tanh(e1_embedded_all)
            e1_embedded = e1_embedded_all[triplets[:, 0]].unsqueeze(1)  # (batch_size, 1, h_dim)
            # 关系的分布式嵌入
            rel_embedded_all = emb_rel[-1]
            rel_embedded = rel_embedded_all[triplets[:, 1]].unsqueeze(1)  # (batch_size, 1, h_dim)
            stacked_inputs = torch.cat([e1_embedded, rel_embedded], 1)  # (batch_size, 2, h_dim)
            stacked_inputs = self.bn0(stacked_inputs)
            x0 = self.inp_drop(stacked_inputs)
            x0 = self.conv1(x0)
            x0 = self.bn1(x0)
            x0 = F.relu(x0)
            x0 = self.feature_map_drop(x0)
            x0 = x0.view(batch_size, -1)  # (batch_size, h_dim*channels)
            x0 = self.fc(x0)  # (batch_size, h_dim)
            x = x0
        if len(embedding) == 2:
            # Number 0
            evolve_emb_0 = embedding[0]
            # 实体的分布式嵌入
            e1_embedded_all = F.normalize(evolve_emb_0) if layer_norm else evolve_emb_0
            e1_embedded_all = F.tanh(e1_embedded_all)
            e1_embedded = e1_embedded_all[triplets[:, 0]].unsqueeze(1)  # (batch_size, 1, h_dim)
            # 关系的分布式嵌入
            rel_embedded_all = emb_rel[-1]
            rel_embedded = rel_embedded_all[triplets[:, 1]].unsqueeze(1)  # (batch_size, 1, h_dim)
            stacked_inputs = torch.cat([e1_embedded, rel_embedded], 1)  # (batch_size, 2, h_dim)
            stacked_inputs = self.bn0(stacked_inputs)
            x0 = self.inp_drop(stacked_inputs)
            x0 = self.conv1(x0)
            x0 = self.bn1(x0)
            x0 = F.relu(x0)
            x0 = self.feature_map_drop(x0)
            x0 = x0.view(batch_size, -1)  # (batch_size, h_dim*channels)
            x0 = self.fc(x0)  # (batch_size, h_dim)
            x0 = x0.unsqueeze(1)  # (batch_size, 1, h_dim)

            # Number 1
            evolve_emb_1 = embedding[1]
            # 实体的分布式嵌入
            e1_embedded_all = F.normalize(evolve_emb_1) if layer_norm else evolve_emb_1
            e1_embedded_all = F.tanh(e1_embedded_all)
            e1_embedded = e1_embedded_all[triplets[:, 0]].unsqueeze(1)  # (batch_size, 1, h_dim)
            # 关系的分布式嵌入
            rel_embedded_all = emb_rel[-1]
            rel_embedded = rel_embedded_all[triplets[:, 1]].unsqueeze(1)  # (batch_size, 1, h_dim)
            stacked_inputs = torch.cat([e1_embedded, rel_embedded], 1)  # (batch_size, 2, h_dim)
            stacked_inputs = self.bn0(stacked_inputs)
            x1 = self.inp_drop(stacked_inputs)
            x1 = self.conv1(x1)
            x1 = self.bn1(x1)
            x1 = F.relu(x1)
            x1 = self.feature_map_drop(x1)
            x1 = x1.view(batch_size, -1)  # (batch_size, h_dim*channels)
            x1 = self.fc(x1)  # (batch_size, h_dim)
            x1 = x1.unsqueeze(1)  # (batch_size, 1, h_dim)

            # 注意力机制
            x_sub = x1  # (batch_size, 1, h_dim)
            x_obj = torch.cat([x0, x1], 1)  # (batch_size, 2, h_dim)
            for layer in self.layers:
                x_sub, x_obj = layer(x_sub, x_obj)  # x_sub: (batch_size, 1, h_dim) x_obj: (batch_size, 2, h_dim)
            x = torch.squeeze(x_sub, 1)  # (batch_size, h_dim)
        if len(embedding) == 3:
            # Number 0
            evolve_emb_0 = embedding[0]
            # 实体的分布式嵌入
            e1_embedded_all = F.normalize(evolve_emb_0) if layer_norm else evolve_emb_0
            e1_embedded_all = F.tanh(e1_embedded_all)
            e1_embedded = e1_embedded_all[triplets[:, 0]].unsqueeze(1) # (batch_size, 1, h_dim)
            # 关系的分布式嵌入
            rel_embedded_all = emb_rel[-1]
            rel_embedded = rel_embedded_all[triplets[:, 1]].unsqueeze(1)  # (batch_size, 1, h_dim)
            stacked_inputs = torch.cat([e1_embedded, rel_embedded], 1)  # (batch_size, 2, h_dim)
            stacked_inputs = self.bn0(stacked_inputs)
            x0 = self.inp_drop(stacked_inputs)
            x0 = self.conv1(x0)
            x0 = self.bn1(x0)
            x0 = F.relu(x0)
            x0 = self.feature_map_drop(x0)
            x0 = x0.view(batch_size, -1) # (batch_size, h_dim*channels)
            x0 = self.fc(x0) # (batch_size, h_dim)
            x0 = x0.unsqueeze(1) # (batch_size, 1, h_dim)

            # Number 1
            evolve_emb_1 = embedding[1]
            # 实体的分布式嵌入
            e1_embedded_all = F.normalize(evolve_emb_1) if layer_norm else evolve_emb_1
            e1_embedded_all = F.tanh(e1_embedded_all)
            e1_embedded = e1_embedded_all[triplets[:, 0]].unsqueeze(1)  # (batch_size, 1, h_dim)
            # 关系的分布式嵌入
            rel_embedded_all = emb_rel[-1]
            rel_embedded = rel_embedded_all[triplets[:, 1]].unsqueeze(1)  # (batch_size, 1, h_dim)
            stacked_inputs = torch.cat([e1_embedded, rel_embedded], 1)  # (batch_size, 2, h_dim)
            stacked_inputs = self.bn0(stacked_inputs)
            x1 = self.inp_drop(stacked_inputs)
            x1 = self.conv1(x1)
            x1 = self.bn1(x1)
            x1 = F.relu(x1)
            x1 = self.feature_map_drop(x1)
            x1 = x1.view(batch_size, -1)  # (batch_size, h_dim*channels)
            x1 = self.fc(x1)  # (batch_size, h_dim)
            x1 = x1.unsqueeze(1)  # (batch_size, 1, h_dim)

            # Number 2
            evolve_emb_2 = embedding[2]
            # 实体的分布式嵌入
            e1_embedded_all = F.normalize(evolve_emb_2) if layer_norm else evolve_emb_2
            e1_embedded_all = F.tanh(e1_embedded_all)
            e1_embedded = e1_embedded_all[triplets[:, 0]].unsqueeze(1)  # (batch_size, 1, h_dim)
            # 关系的分布式嵌入
            rel_embedded_all = emb_rel[-1]
            rel_embedded = rel_embedded_all[triplets[:, 1]].unsqueeze(1)  # (batch_size, 1, h_dim)
            stacked_inputs = torch.cat([e1_embedded, rel_embedded], 1)  # (batch_size, 2, h_dim)
            stacked_inputs = self.bn0(stacked_inputs)
            x2 = self.inp_drop(stacked_inputs)
            x2 = self.conv1(x2)
            x2 = self.bn1(x2)
            x2 = F.relu(x2)
            x2 = self.feature_map_drop(x2)
            x2 = x2.view(batch_size, -1)  # (batch_size, h_dim*channels)
            x2 = self.fc(x2)  # (batch_size, h_dim)
            x2 = x2.unsqueeze(1)  # (batch_size, 1, h_dim)

            # 注意力机制
            x_sub = x2 # (batch_size, 1, h_dim)
            x_obj = torch.cat([x0, x1, x2], 1) # (batch_size, 3, h_dim)
            for layer in self.layers:
                x_sub, x_obj = layer(x_sub, x_obj) # x_sub: (batch_size, 1, h_dim) x_obj: (batch_size, 3, h_dim)
            x = torch.squeeze(x_sub, 1)  # (batch_size, h_dim)

        x = self.hidden_drop(x)
        if batch_size > 1:
            x = self.bn2(x)
        x = F.relu(x) # (batch_size, h_dim)
        if partial_embeding is None:
            x = torch.mm(x, e1_embedded_all.transpose(1, 0)) # x: (batch_size, h_dim), e1_embedding_all.trans: (h_dim, num_ents)
        else:
            x = torch.mm(x, partial_embeding.transpose(1, 0))

        self.output = x
        # 基于历史频次和历史mask的统计，对score进行最后的调整
        if history_tail_seq is not None and one_hot_tail_seq is not None:
            q_score = self.output
            history_mask = torch.tensor(np.array(one_hot_tail_seq.cpu() == 0, dtype=float) * (-100))
            if use_cuda:
                history_mask = history_mask.to(self.gpu)
            temp_mask = torch.tensor(np.array(one_hot_tail_seq.cpu() == 0, dtype=float) * (-1e9))  # 等于0变成-1e9，等于1变成0
            if use_cuda:
                temp_mask = temp_mask.to(self.gpu)
            history_frequency = F.softmax(history_tail_seq + temp_mask, dim=1) * 0.5  # (batch_size, output_dim)
            final_score = history_mask + history_frequency + q_score
        else:
            final_score = self.output
        return final_score

class MultiHeadAttention(nn.Module):
    def __init__(self, h_dim, d_k, d_v, n_heads, gpu):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(h_dim, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(h_dim, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(h_dim, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, h_dim, bias=False)
        self.h_dim = h_dim
        self.d_k = d_k
        self.d_v = d_v # d_k = d_v = d_q
        self.n_heads = n_heads
        self.gpu = gpu
    def forward(self, input_Q, input_K, input_V): # enc_query_inputs, enc_history_inputs, enc_history_inputs
        '''
        input_Q: (batch_size, 1, h_dim)
        input_K: (batch_size, 3, h_dim)
        input_V: (batch_size, 3, h_dim)
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2) # Q: (batch_size, n_heads, 1, d_k)
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2) # K: (batch_size, n_heads, 3, d_k)
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2) # V: (batch_size, n_heads, 3, d_v)

        context = ScaledDotProductAttention(self.d_k)(Q, K, V)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v) # context: (batch_size, 1, n_heads * d_v)
        output = self.fc(context) # (batch_size, 1, h_dim)
        return nn.LayerNorm(self.h_dim).to(self.gpu)(output + residual) # (batch_size, 1, h_dim)

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
    def forward(self, Q, K, V):
        '''
        Q: (batch_size, n_heads, 1, d_k)
        K: (batch_size, n_heads, 3, d_k)
        V: (batch_size, n_heads, 3, d_v)
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k) # scores : (batch_size, n_heads, 1, 3)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V) # (batch_size, n_heads, 1, d_v)
        return context # (batch_size, n_heads, 1, d_v)

class FeedForwardNet(nn.Module):
    def __init__(self, h_dim, d_ff, gpu):
        super(FeedForwardNet, self).__init__()
        self.h_dim = h_dim
        self.fc = nn.Sequential(
            nn.Linear(h_dim, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, h_dim, bias=False)
        )
        self.gpu = gpu
    def forward(self, inputs):
        '''
        inputs: (batch_size, 1, h_dim)
        '''
        residual = inputs
        output = self.fc(inputs)
        # nn.LayerNorm(x)会对输入的最后一维进行归一化, x需要和输入的最后一维一样大
        return nn.LayerNorm(self.h_dim).to(self.gpu)(output + residual) # (batch_size, 1, h_dim_2)

class DecoderLayer(nn.Module):
    def __init__(self, h_dim, d_k, d_v, n_heads, d_ff, gpu):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(h_dim, d_k, d_v, n_heads, gpu)
        self.pos_ffn = FeedForwardNet(h_dim, d_ff, gpu)
    def forward(self, dec_query_inputs, dec_history_inputs):
        '''
        dec_query_inputs: (batch_size, 1, h_dim)
        dec_history_iutputs: (batch_size, 3, h_dim) 由dec_inputs得到
        '''
        # enc_query_inputs生成Q矩阵, enc_history_inputs生成K, V矩阵
        dec_outputs = self.dec_self_attn(dec_query_inputs, dec_history_inputs, dec_history_inputs) # (batch_size, 1, h_dim)
        dec_query_outputs = self.pos_ffn(dec_outputs) # (batch_size, 1, h_dim)
        dec_history_outputs = dec_history_inputs
        return dec_query_outputs, dec_history_outputs # dec_query_outputs: (batch_size, 1, h_dim) dec_history_outputs: (batch_size, 3, h_dim)