import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from rgcn.layers import RGCNBlockLayer as RGCNLayer
from rgcn.layers import UnionRGCNLayer, RGCNBlockLayer
from src.model import BaseRGCN
from src.decoder import ConvTransE

import sys
import scipy.sparse as sp
sys.path.append("..")

class RGCNCell(BaseRGCN):
    def build_hidden_layer(self, idx): # 实现了BaseRGCN中的build_hidden_layer
        act = F.rrelu
        if idx:
            self.num_basis = 0
        print("activate function: {}".format(act))
        if self.skip_connect:
            sc = False if idx == 0 else True
        else:
            sc = False
        if self.encoder_name == "uvrgcn": # 2层的UnionRGCNLayer
            # num_rels*2
            return UnionRGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                             activation=act, dropout=self.dropout, self_loop=self.self_loop, skip_connect=sc, rel_emb=self.rel_emb)
        else:
            raise NotImplementedError


    def forward(self, g, init_ent_emb, init_rel_emb):
        # g: 当前历史子图; self.h: node嵌入 (num_ents, h_dim); [self.h_0, self.h_0]: 边的嵌入
        if self.encoder_name == "uvrgcn":
            node_id = g.ndata['id'].squeeze() # node id
            g.ndata['h'] = init_ent_emb[node_id] # node embedding
            x, r = init_ent_emb, init_rel_emb
            for i, layer in enumerate(self.layers): # n_layers = 2 两层的UnionRGCNLayer
                layer(g, [], r[i]) # g: 当前历史子图; r[i]: self.h_0 (num_rels*2, h_dim) 更新了两轮的g.ndata['h']
            return g.ndata.pop('h') # 返回了图中更新的node embedding
        else:
            if self.features is not None:
                print("----------------Feature is not None, Attention ------------")
                g.ndata['id'] = self.features
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            if self.skip_connect:
                prev_h = []
                for layer in self.layers:
                    prev_h = layer(g, prev_h)
            else:
                for layer in self.layers:
                    layer(g, [])
            return g.ndata.pop('h')



class RecurrentRGCN(nn.Module):
    def __init__(self, decoder_name, encoder_name, num_ents, num_rels, num_static_rels, num_words, h_dim, opn, sequence_len, num_bases=-1, num_basis=-1,
                 num_hidden_layers=1, dropout=0, self_loop=False, skip_connect=False, layer_norm=False, input_dropout=0,
                 hidden_dropout=0, feat_dropout=0, aggregation='cat', use_static=False,
                 entity_prediction=False, use_cuda=False, gpu = 0, analysis=False):
        super(RecurrentRGCN, self).__init__()

        self.decoder_name = decoder_name # convtranse
        self.encoder_name = encoder_name # uvrgcn
        self.num_rels = num_rels
        self.num_ents = num_ents
        self.opn = opn
        self.num_words = num_words
        self.num_static_rels = num_static_rels
        self.sequence_len = sequence_len
        self.h_dim = h_dim
        self.layer_norm = layer_norm
        self.h = None
        self.run_analysis = analysis
        self.aggregation = aggregation
        self.relation_evolve = False
        self.use_static = use_static
        self.entity_prediction = entity_prediction
        self.emb_rel = None
        self.gpu = gpu

        self.w1 = torch.nn.Parameter(torch.Tensor(self.h_dim, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.w1)

        self.w2 = torch.nn.Parameter(torch.Tensor(self.h_dim, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.w2)

        self.emb_rel = torch.nn.Parameter(torch.Tensor(self.num_rels * 2, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.emb_rel)

        self.dynamic_emb = torch.nn.Parameter(torch.Tensor(num_ents, h_dim), requires_grad=True).float() # 所有实体的进化嵌入
        torch.nn.init.normal_(self.dynamic_emb)

        self.loss_e = torch.nn.CrossEntropyLoss()

        self.rgcn = RGCNCell(num_ents,
                             h_dim,
                             h_dim,
                             num_rels * 2,
                             num_bases,
                             num_basis,
                             num_hidden_layers,
                             dropout,
                             self_loop,
                             skip_connect,
                             encoder_name,
                             self.opn,
                             self.emb_rel,
                             use_cuda,
                             analysis)

        # self.time_gate_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))
        # nn.init.xavier_uniform_(self.time_gate_weight, gain=nn.init.calculate_gain('relu'))
        # self.time_gate_bias = nn.Parameter(torch.Tensor(h_dim))
        # nn.init.zeros_(self.time_gate_bias)

        # GRU cell for relation evolving
        self.relation_cell_1 = nn.GRUCell(self.h_dim*2, self.h_dim)

        # decoder
        if decoder_name == "convtranse":
            self.decoder_ob = ConvTransE(num_entities=num_ents, embedding_dim=h_dim, gpu=gpu, layer_norm=layer_norm,
                                         input_dropout=input_dropout, hidden_dropout=hidden_dropout, feature_map_dropout=feat_dropout)
        else:
            raise NotImplementedError 

    def forward(self, g_list, static_graph, use_cuda):
        gate_list = []
        degree_list = []

        self.h = F.normalize(self.dynamic_emb) if self.layer_norm else self.dynamic_emb[:, :]
        static_emb = None

        history_embs = []
        rel_embs = []

        for i, g in enumerate(g_list): # 对于每一个历史子图
            g = g.to(self.gpu)
            # self.h: (num_ents, h_dim); g.r_to_e: 子图g中和每一条边r相关的node列表，按照g.uniq_r中记录边的顺序排列
            temp_e = self.h[g.r_to_e] # 取出r_to_e列表中node的embedding
            # x_input: (num_rels*2, h_dim) 所有边的embedding
            x_input = torch.zeros(self.num_rels * 2, self.h_dim).float().cuda() if use_cuda else torch.zeros(self.num_rels * 2, self.h_dim).float()
            # g.r_len: 记录和边r相关的node在r_to_e列表中的idx范围，也与uniq_r中边的顺序保持一致 [(0, 4), (4, 9), ...]
            # g.uniq_r: 在当前时间戳内出现的所有的边(包括反向边) [r0, r1, ..., r0', r1', ...]
            for span, r_idx in zip(g.r_len, g.uniq_r):
                x = temp_e[span[0]:span[1],:] # 取出与关系r相关的所有node的embedding
                x_mean = torch.mean(x, dim=0, keepdim=True) # (1, h_dim), 将当前r_idx相关的所有node的embedding求均值
                x_input[r_idx] = x_mean
            # emb_rel: (num_rels*2, h_dim)边的嵌入 x_input: (num_rels*2, h_dim)聚合边的相关node的边的嵌入
            # 处理边的
            if i == 0: # 第一个历史子图
                x_input = torch.cat((self.emb_rel, x_input), dim=1) # (num_rels*2, h_dim*2)
                self.h_0 = self.relation_cell_1(x_input, self.emb_rel)    # 第1层输入
                self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
                rel_embs.append(self.h_0)
            else:
                x_input = torch.cat((self.emb_rel, x_input), dim=1) # (num_rels*2, h_dim*2)
                self.h_0 = self.relation_cell_1(x_input, self.h_0)  # 第2层输出==下一时刻第一层输入
                self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0 # self.h_0: (num_rels*2, h_dim)
                rel_embs.append(self.h_0)
            # g: 当前历史子图; self.h: node嵌入 (num_ents, h_dim); [self.h_0, self.h_0]: 边的嵌入 (num_rels*2, h_dim) 因为有2层，所以传入两个输入
            current_h = self.rgcn.forward(g, self.h, [self.h_0, self.h_0]) # 返回了图中更新的node embedding
            current_h = F.normalize(current_h) if self.layer_norm else current_h
            # time_weight = F.sigmoid(torch.mm(self.h, self.time_gate_weight) + self.time_gate_bias)
            # self.h = time_weight * current_h + (1-time_weight) * self.h # current_h: 当前历史子图的node embedding; 贯穿历史序列的整体图谱node embedding
            self.h = current_h # self.h: (num_ents, h_dim)
            history_embs.append(self.h) # 每一个历史子图的node embedding
        return history_embs, static_emb, rel_embs, gate_list, degree_list


    def predict(self, test_graph, num_rels, static_graph, history_tail_seq, one_hot_tail_seq, test_triplets, use_cuda):
        '''
        :param test_graph:
        :param num_rels: 原始关系数目
        :param static_graph:
        :param history_tail_seq:
        :param one_hot_tail_seq
        :param test_triplets: 一个时间戳内的所有事实 [[s, r, o], [], ...] (num_triples_time, 3)
        :param use_cuda:
        :return:
        '''
        with torch.no_grad():
            inverse_test_triplets = test_triplets[:, [2, 1, 0]]
            inverse_test_triplets[:, 1] = inverse_test_triplets[:, 1] + num_rels  # 将逆关系换成逆关系的id
            all_triples = torch.cat((test_triplets, inverse_test_triplets)) # (batch_size, 3)
            
            evolve_embs, _, r_embs, _, _ = self.forward(test_graph, static_graph, use_cuda)
            # embedding = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1] # embedding: (num_ents, h_dim); r_emb: (num_rels*2, h_dim)

            score = self.decoder_ob.forward(evolve_embs, r_embs, all_triples, history_tail_seq, one_hot_tail_seq, self.layer_norm, use_cuda, mode="test") # all_triples: 包含反关系的三元组二维张量

            return all_triples, score # (batch_size, 3) (batch_size, num_ents)


    def get_loss(self, glist, triples, static_graph, history_tail_seq, one_hot_tail_seq, use_cuda):
        """
        还需传入当前时间戳下的所有事实在各个历史子图中的历史重复事实列表
        :param glist: 历史子图列表
        :param triplets: 当前时间戳下的所有事实，一个时间戳内的所有事实三元组
        :param static_graph:
        :param history_tail_seq:
        :param one_hot_tail_seq
        :param use_cuda:
        :return:
        """
        loss_ent = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)

        inverse_triples = triples[:, [2, 1, 0]]
        inverse_triples[:, 1] = inverse_triples[:, 1] + self.num_rels
        all_triples = torch.cat([triples, inverse_triples])
        all_triples = all_triples.to(self.gpu)

        evolve_embs, static_emb, r_embs, _, _ = self.forward(glist, static_graph, use_cuda) # evolve_embs, static_emb, r_emb在gpu上
        # pre_emb = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]

        if self.entity_prediction:
            scores_ob = self.decoder_ob.forward(evolve_embs, r_embs, all_triples, history_tail_seq, one_hot_tail_seq, self.layer_norm, use_cuda).view(-1, self.num_ents)
            loss_ent += self.loss_e(scores_ob, all_triples[:, 2])
     
        return loss_ent
