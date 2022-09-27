"""
Utility functions for link prediction
Most code is adapted from authors' implementation of RGCN link prediction:
https://github.com/MichSchli/RelationPrediction

"""
import numpy as np
import torch
import dgl
from tqdm import tqdm
import rgcn.knowledge_graph as knwlgrh
from collections import defaultdict


#######################################################################
#
# Utility function for building training and testing graphs
#
#######################################################################

def sort_and_rank(score, target):
    '''
    :param score: (batch_size, num_ents) num_ents中score下标与实体编号相对应
    :param target: (batch_size, 1)
    '''
    # 对于被time-aware filter掉的实体得分被赋为-inf，排名在最后不参与排名
    _, indices = torch.sort(score, dim=1, descending=True) # indices: 实体编号按照score从大到小排列
    indices = torch.nonzero(indices == target.view(-1, 1)) # 分别找到这一个batch_size的事实中的ground truth排第几
    indices = indices[:, 1].view(-1) # (batch_size) 每一个值对应一个事实三元组的ground truth排名
    return indices


#TODO filer by groud truth in the same time snapshot not all ground truth
def sort_and_rank_time_filter(batch_a, batch_r, score, target, total_triplets):
    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))
    for i in range(len(batch_a)):
        ground = indices[i]
    indices = indices[:, 1].view(-1)
    return indices


def sort_and_rank_filter(batch_a, batch_r, score, target, all_ans):
    for i in range(len(batch_a)):
        ans = target[i]
        b_multi = list(all_ans[batch_a[i].item()][batch_r[i].item()])
        ground = score[i][ans]
        score[i][b_multi] = 0
        score[i][ans] = ground
    _, indices = torch.sort(score, dim=1, descending=True)  # indices : [B, number entity]
    indices = torch.nonzero(indices == target.view(-1, 1))  # indices : [B, 2] 第一列递增， 第二列表示对应的答案实体id在每一行的位置
    indices = indices[:, 1].view(-1)
    return indices


def filter_score(test_triples, score, all_ans):
    '''
    :param test_triples: (batch_size, 3)
    :param score: (batch_size, num_ents)
    :param all_ans: {e1: {rel: set(e2)}, e2: {rel+num_rels: set(e1)}, ...} 当前时间戳下的事实出现情况（包括反关系）
    '''
    if all_ans is None:
        return score
    test_triples = test_triples.cpu() # (batch_size, 3)
    for _, triple in enumerate(test_triples): # 对于一个batch中的每一个事实三元组
        h, r, t = triple
        ans = list(all_ans[h.item()][r.item()]) # 当前h和r，预测t的情况下应该filtered掉的实体
        ans.remove(t.item()) # ground truth需要参与排名，不能被filter
        ans = torch.LongTensor(ans)
        score[_][ans] = -10000000 # 对应被filter掉的实体的得分转换为-inf值
    return score

def filter_score_r(test_triples, score, all_ans):
    if all_ans is None:
        return score
    test_triples = test_triples.cpu()
    for _, triple in enumerate(test_triples):
        h, r, t = triple
        ans = list(all_ans[h.item()][t.item()])
        # print(h, r, t)
        # print(ans)
        ans.remove(r.item())
        ans = torch.LongTensor(ans)
        score[_][ans] = -10000000  #
    return score


def r2e(triplets, num_rels): # triplets: [[s, r, o], [s, r, o], ...]
    src, rel, dst = triplets.transpose()
    # get all relations
    uniq_r = np.unique(rel) # 从小到大排列
    uniq_r = np.concatenate((uniq_r, uniq_r+num_rels)) # 在当前时间戳内出现的所有的边
    # generate r2e
    r_to_e = defaultdict(set) # 获得和每一条边相关的节点
    for j, (src, rel, dst) in enumerate(triplets): # 对于时间戳内的每一个事实三元组
        r_to_e[rel].add(src)
        r_to_e[rel].add(dst)
        r_to_e[rel+num_rels].add(src)
        r_to_e[rel+num_rels].add(dst)
    r_len = []
    e_idx = []
    idx = 0
    for r in uniq_r: # 对于在该时间戳内出现的每一条边
        r_len.append((idx,idx+len(r_to_e[r]))) # 记录和边r相关的node的idx范围
        e_idx.extend(list(r_to_e[r])) # 和边r相关的node列表
        idx += len(r_to_e[r])
    return uniq_r, r_len, e_idx


def build_sub_graph(num_nodes, num_rels, triples, use_cuda, gpu):
    '''
    :param num_nodes:
    :param num_rels:
    :param triples: 一个历史时间戳的所有事实三元组 [[s, r, o], [s, r, o], ...]
    :param use_cuda:
    :param gpu:
    :return:
    '''
    def comp_deg_norm(g):
        in_deg = g.in_degrees(range(g.number_of_nodes())).float() # 图中每一个节点的入度
        in_deg[torch.nonzero(in_deg == 0).view(-1)] = 1 # 入度为0的赋值为1
        norm = 1.0 / in_deg # 归一化操作 1/入度
        return norm

    src, rel, dst = triples.transpose() # (3 * 事实个数)
    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src)) # knowledge graph 无向图, 加入反关系
    rel = np.concatenate((rel, rel + num_rels)) # 关系+反关系

    g = dgl.DGLGraph()
    g.add_nodes(num_nodes) # 加入所有节点
    g.add_edges(src, dst) # 加入所有边
    norm = comp_deg_norm(g) # 对一个子图中的所有节点进行归一化
    node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1) # [0, num_nodes)
    g.ndata.update({'id': node_id, 'norm': norm.view(-1, 1)}) # shape都为(num_nodes, 1)
    g.apply_edges(lambda edges: {'norm': edges.dst['norm'] * edges.src['norm']}) # 更新边, 边的归一化系数为头尾节点的归一化系数相乘
    g.edata['type'] = torch.LongTensor(rel) # 边的类型数据

    uniq_r, r_len, r_to_e = r2e(triples, num_rels) # uniq_r: 在当前时间戳内出现的所有的边(包括反向边)；r_len: 记录和边r相关的node的idx范围; e_idx: 和边r相关的node列表
    g.uniq_r = uniq_r # 在当前时间戳内出现的所有的边(包括反向边)
    g.r_to_e = r_to_e # 和边r相关的node列表，按照uniq_r中记录边的顺序排列
    g.r_len = r_len # 记录和边r相关的node在r_to_e列表中的idx范围，也与uniq_r中边的顺序保持一致
    if use_cuda:
        g.to(gpu)
        g.r_to_e = torch.from_numpy(np.array(r_to_e))
    return g

def get_total_rank(test_triples, score, all_ans, eval_bz):
    '''
    :param test_triples: (num_triples_time*2, 3) num_triples_time*2（一个时间戳中的所有事实三元组（包括反关系））
    :param score: (num_triples_time*2, num_ents)
    :param all_ans: {e1: {rel: set(e2)}, e2: {rel+num_rels: set(e1)}, ...} 当前时间戳下的事实出现情况（包括反关系）
    :param eval_bz: 1000
    :return:
    '''
    num_triples = len(test_triples) # 当前测试时间戳下的所有事实三元组数量
    n_batch = (num_triples + eval_bz - 1) // eval_bz # 一个时间戳内的处理batch数量
    rank = []
    filter_rank = []
    for idx in range(n_batch): # 对于一个时间戳内的每一个batch
        batch_start = idx * eval_bz
        batch_end = min(num_triples, (idx + 1) * eval_bz)
        triples_batch = test_triples[batch_start:batch_end, :] # (tensor)(batch_size, 3)
        score_batch = score[batch_start:batch_end, :] # (batch_size, num_ents)
        target = test_triples[batch_start:batch_end, 2] # ground truth (batch_size, 1)
        rank.append(sort_and_rank(score_batch, target)) # (batch_size) 一个batch中每一个事实三元组的ground truth排名

        filter_score_batch = filter_score(triples_batch, score_batch, all_ans) # (batch_size, 3) (batch_size, num_ents)
        filter_rank.append(sort_and_rank(filter_score_batch, target))

    rank = torch.cat(rank)
    filter_rank = torch.cat(filter_rank)
    rank += 1 # change to 1-indexed
    filter_rank += 1
    mrr = torch.mean(1.0 / rank.float())
    filter_mrr = torch.mean(1.0 / filter_rank.float())
    return filter_mrr.item(), mrr.item(), rank, filter_rank


def stat_ranks(rank_list, method):
    hits = [1, 3, 10]
    total_rank = torch.cat(rank_list)

    mrr = torch.mean(1.0 / total_rank.float())
    print("MRR ({}): {:.6f}".format(method, mrr.item()))
    for hit in hits:
        avg_count = torch.mean((total_rank <= hit).float())
        print("Hits ({}) @ {}: {:.6f}".format(method, hit, avg_count.item()))
    return mrr


def flatten(l):
    flatten_l = []
    for c in l:
        if type(c) is list or type(c) is tuple:
            flatten_l.extend(flatten(c))
        else:
            flatten_l.append(c)
    return flatten_l

def UnionFindSet(m, edges):
    """

    :param m:
    :param edges:
    :return: union number in a graph
    """
    roots = [i for i in range(m)]
    rank = [0 for i in range(m)]
    count = m

    def find(member):
        tmp = []
        while member != roots[member]:
            tmp.append(member)
            member = roots[member]
        for root in tmp:
            roots[root] = member
        return member

    for i in range(m):
        roots[i] = i
    # print ufs.roots
    for edge in edges:
        print(edge)
        start, end = edge[0], edge[1]
        parentP = find(start)
        parentQ = find(end)
        if parentP != parentQ:
            if rank[parentP] > rank[parentQ]:
                roots[parentQ] = parentP
            elif rank[parentP] < rank[parentQ]:
                roots[parentP] = parentQ
            else:
                roots[parentQ] = parentP
                rank[parentP] -= 1
            count -= 1
    return count

def append_object(e1, e2, r, d):
    if not e1 in d:
        d[e1] = {}
    if not r in d[e1]:
        d[e1][r] = set()
    d[e1][r].add(e2)


def add_subject(e1, e2, r, d, num_rel): # (subject, r, object) 统计从每一个obj的obj->sub的反向关系下与之相邻接的sub集合: {e2: {r+num_rel: set(e1)}, ...}
    if not e2 in d: # 如果all_ans字典中不存在以e2(object)为键的键值对
        d[e2] = {}
    if not r+num_rel in d[e2]: # r: sub->obj; r+num+rel: obj->sub
        d[e2][r+num_rel] = set()
    d[e2][r+num_rel].add(e1)


def add_object(e1, e2, r, d, num_rel): # 统计从每一个sub的sub->obj的关系下与之相邻接的obj集合: {e1: {r: set(e2)}, ...}
    if not e1 in d:
        d[e1] = {}
    if not r in d[e1]:
        d[e1][r] = set()
    d[e1][r].add(e2)


def load_all_answers(total_data, num_rel):
    # store subjects for all (rel, object) queries and
    # objects for all (subject, rel) queries
    all_subjects, all_objects = {}, {}
    for line in total_data:
        s, r, o = line[: 3]
        add_subject(s, o, r, all_subjects, num_rel=num_rel)
        add_object(s, o, r, all_objects, num_rel=0)
    return all_objects, all_subjects


def load_all_answers_for_filter(total_data, num_rel): # 一个时间戳内包含的所有事实：[[s, r, o], [], ...]
    # store subjects for all (rel, object) queries and
    # objects for all (subject, rel) queries
    all_ans = {} # {e1: {e2: set()}，...}
    for line in total_data: # 针对一个时间戳内的每一个事实
        s, r, o = line[: 3]
        add_subject(s, o, r, all_ans, num_rel=num_rel)
        add_object(s, o, r, all_ans, num_rel=0)
    return all_ans # {e1: {rel: set(e2)}, e2: {rel+num_rels: set(e1)}, ...}


def load_all_answers_for_time_filter(total_data, num_rels, num_nodes):
    all_ans_list = []
    all_snap = split_by_time(total_data) # 按时间戳划分的array：[[[s, r, o], [], ...], [], ...]
    for snap in all_snap: # 对于每一个时间戳 snap包含一个时间戳内的所有事实 [[s, r, o], [], ...]
        all_ans_t = load_all_answers_for_filter(snap, num_rels)
        all_ans_list.append(all_ans_t) # 按照时间戳排序
    return all_ans_list

def split_by_time(data):
    snapshot_list = []
    snapshot = []
    snapshots_num = 0
    latest_t = 0
    for i in range(len(data)): # np.array([[s, r, o, time], []...])
        t = data[i][3]
        train = data[i]
        # latest_t表示读取的上一个三元组发生的时刻，要求数据集中的三元组是按照时间发生顺序排序的
        if latest_t != t: # 同一时刻发生的三元组，如果时间戳发生变化，代表进入下一个时间戳
            # show snapshot
            latest_t = t
            if len(snapshot):
                snapshot_list.append(np.array(snapshot).copy()) # 将上一个时间戳的所有事实加入snapshot_list
                snapshots_num += 1
            snapshot = []
        snapshot.append(train[:3])
    if len(snapshot) > 0:
        snapshot_list.append(np.array(snapshot).copy()) # 加入最后一个snapshot
        snapshots_num += 1

    union_num = [1]
    nodes = []
    rels = []
    for snapshot in snapshot_list: # 对于每一个时间戳中的所有事实 np.array([[s, r, o], [], ...])
        uniq_v, edges = np.unique((snapshot[:,0], snapshot[:,2]), return_inverse=True) # uniq_v: 一个时间戳内从小到大排序的非重复实体列表np.array；edges: relabel的[头实体, 尾实体]，采用uniq_v的index
        uniq_r = np.unique(snapshot[:,1])
        edges = np.reshape(edges, (2, -1)) # 重新组织为头实体->尾实体的形式
        nodes.append(len(uniq_v)) # 每一个时间戳内出现过的实体数目
        rels.append(len(uniq_r)*2) # 所有无向edges，一正一反，每一个时间戳内出现过的关系数目
    print("# Sanity Check:  ave node num : {:04f}, ave rel num : {:04f}, snapshots num: {:04d}, max edges num: {:04d}, min edges num: {:04d}, max union rate: {:.4f}, min union rate: {:.4f}"
          .format(np.average(np.array(nodes)), np.average(np.array(rels)), len(snapshot_list), max([len(_) for _ in snapshot_list]), min([len(_) for _ in snapshot_list]), max(union_num), min(union_num)))
    return snapshot_list # 按时间戳划分的array：[[[s, r, o], [], ...], [], ...]

def add_inverse_rel(data, num_rel): # [[s, r, o], [], ...]
    inverse_triples = np.array([[o, r+num_rel, s] for s, r, o in data])
    triples_w_inverse = np.concatenate((data, inverse_triples))
    return triples_w_inverse

def slide_list(snapshots, k=1):
    """
    :param k: padding K history for sequence stat
    :param snapshots: all snapshot
    :return:
    """
    k = k  # k=1 需要取长度k的历史，在加1长度的label
    if k > len(snapshots):
        print("ERROR: history length exceed the length of snapshot: {}>{}".format(k, len(snapshots)))
    for _ in tqdm(range(len(snapshots)-k+1)):
        yield snapshots[_: _+k]



def load_data(dataset, bfs_level=3, relabel=False):
    if dataset in ['aifb', 'mutag', 'bgs', 'am']:
        return knwlgrh.load_entity(dataset, bfs_level, relabel)
    elif dataset in ['FB15k', 'wn18', 'FB15k-237']:
        return knwlgrh.load_link(dataset)
    elif dataset in ['ICEWS18', 'ICEWS14', "GDELT", "SMALL", "ICEWS14s", "ICEWS05-15","YAGO",
                     "WIKI"]:
        return knwlgrh.load_from_local("../data", dataset)
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))

def construct_snap(test_triples, num_nodes, num_rels, final_score, topK):
    sorted_score, indices = torch.sort(final_score, dim=1, descending=True)
    top_indices = indices[:, :topK]
    predict_triples = []
    for _ in range(len(test_triples)):
        for index in top_indices[_]:
            h, r = test_triples[_][0], test_triples[_][1]
            if r < num_rels:
                predict_triples.append([test_triples[_][0], r, index])
            else:
                predict_triples.append([index, r-num_rels, test_triples[_][0]])

    # 转化为numpy array
    predict_triples = np.array(predict_triples, dtype=int)
    return predict_triples

def construct_snap_r(test_triples, num_nodes, num_rels, final_score, topK):
    sorted_score, indices = torch.sort(final_score, dim=1, descending=True)
    top_indices = indices[:, :topK]
    predict_triples = []
    # for _ in range(len(test_triples)):
    #     h, r = test_triples[_][0], test_triples[_][1]
    #     if (sorted_score[_][0]-sorted_score[_][1])/sorted_score[_][0] > 0.3:
    #         if r < num_rels:
    #             predict_triples.append([h, r, indices[_][0]])

    for _ in range(len(test_triples)):
        for index in top_indices[_]:
            h, t = test_triples[_][0], test_triples[_][2]
            if index < num_rels:
                predict_triples.append([h, index, t])
                #predict_triples.append([t, index+num_rels, h])
            else:
                predict_triples.append([t, index-num_rels, h])
                #predict_triples.append([t, index-num_rels, h])

    # 转化为numpy array
    predict_triples = np.array(predict_triples, dtype=int)
    return predict_triples


def dilate_input(input_list, dilate_len):
    dilate_temp = []
    dilate_input_list = []
    for i in range(len(input_list)):
        if i % dilate_len == 0 and i:
            if len(dilate_temp):
                dilate_input_list.append(dilate_temp)
                dilate_temp = []
        if len(dilate_temp):
            dilate_temp = np.concatenate((dilate_temp, input_list[i]))
        else:
            dilate_temp = input_list[i]
    dilate_input_list.append(dilate_temp)
    dilate_input_list = [np.unique(_, axis=0) for _ in dilate_input_list]
    return dilate_input_list

def emb_norm(emb, epo=0.00001):
    x_norm = torch.sqrt(torch.sum(emb.pow(2), dim=1))+epo
    emb = emb/x_norm.view(-1,1)
    return emb

def shuffle(data, labels):
    shuffle_idx = np.arange(len(data))
    np.random.shuffle(shuffle_idx)
    relabel_output = data[shuffle_idx]
    labels = labels[shuffle_idx]
    return relabel_output, labels


def cuda(tensor):
    if tensor.device == torch.device('cpu'):
        return tensor.cuda()
    else:
        return tensor


def soft_max(z):
    t = np.exp(z)
    a = np.exp(z) / np.sum(t)
    return a
