import os
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
import argparse
# python get_history_record.py --d YAGO
# python get_history_record.py --d WIKI
# python get_history_record.py --d ICEWS18
# python get_history_record.py --d ICEWS14t
# python get_history_record.py --d GDELT
parser = argparse.ArgumentParser(description='Config')
parser.add_argument("-d", "--dataset", type=str, required=True,
                        help="dataset to use")
args = parser.parse_args()
print(args)


def load_quadruples(inPath, fileName, num_r):
    quadrupleList = []
    times = set()
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3])
            quadrupleList.append([head, rel, tail, time])
            times.add(time)
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3])
            quadrupleList.append([tail, rel + num_r, head, time])
    times = list(times)
    times.sort() # 从小到大排序
    return np.asarray(quadrupleList), np.asarray(times)

def get_total_number(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1])

def get_data_with_t(data, tim):
    triples = [[quad[0], quad[1], quad[2]] for quad in data if quad[3] == tim]
    return np.array(triples)
num_e, num_r = get_total_number('./data/{}'.format(args.dataset), 'stat.txt') # 10623 10
# [[h, r, t, time], ...] 训练集时间戳集合（从小到大）
train_data, train_times = load_quadruples('./data/{}'.format(args.dataset), 'train.txt', num_r) # 训练集事实, 时间戳（从小到大）
dev_data, dev_times = load_quadruples('./data/{}'.format(args.dataset), 'valid.txt', num_r)  # 验证集事实, 时间戳（从小到大）
test_data, test_times = load_quadruples('./data/{}'.format(args.dataset), 'test.txt', num_r)  # 测试集事实, 时间戳（从小到大）
all_data = np.concatenate((train_data, dev_data, test_data), axis=0) # 包含反关系
all_times = np.concatenate((train_times, dev_times, test_times))
# 实体数目 关系数目

save_dir_obj = './data/{}/history_seq/'.format(args.dataset)

def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

mkdirs(save_dir_obj)

# get object_entities 关系矩阵
num_r_2 = num_r * 2
row = all_data[:, 0] * num_r_2 + all_data[:, 1] # (n,) 一维
col_rel = all_data[:, 1]
d_ = np.ones(len(row))
tail_rel = sp.csr_matrix((d_, (row, col_rel)), shape=(num_e * num_r_2, num_r_2)) # 关系矩阵的压缩存储，关系矩阵的值就是边
sp.save_npz('./data/{}/history_seq/h_r_seq_rel.npz'.format(args.dataset), tail_rel)

# for tim in tqdm(train_times): # 对于每一个训练集的时间戳（从小到大）
#     # 取出该时间戳下的所有训练集中的事实四元组 设为(n, 4)
#     train_new_data = np.array([[quad[0], quad[1], quad[2], quad[3]] for quad in train_data if quad[3] == tim])
#     # get object_entities
#     row = train_new_data[:, 0] * num_r_2 + train_new_data[:, 1] # (n,) 一维
#     col = train_new_data[:, 2]
#     d = np.ones(len(row))
#     tail_seq = sp.csr_matrix((d, (row, col)), shape=(num_e * num_r_2, num_e)) # 历史矩阵的压缩存储
#     sp.save_npz('./data/{}/history_seq/train_h_r_history_seq_{}.npz'.format(args.dataset, tim), tail_seq) # tim信息在文件名里
#     sp.save_npz('./data/{}/history_seq/train_dev_h_r_history_seq_{}.npz'.format(args.dataset, tim), tail_seq)
#     sp.save_npz('./data/{}/history_seq/train_dev_test_h_r_history_seq_{}.npz'.format(args.dataset, tim), tail_seq)
# for tim in tqdm(dev_times): # 对于每一个训练集的时间戳（从小到大）
#     # 取出该时间戳下的所有训练集中的事实四元组 设为(n, 4)
#     dev_new_data = np.array([[quad[0], quad[1], quad[2], quad[3]] for quad in dev_data if quad[3] == tim])
#     # get object_entities
#     row = dev_new_data[:, 0] * num_r_2 + dev_new_data[:, 1] # (n,) 一维
#     col = dev_new_data[:, 2]
#     d = np.ones(len(row))
#     tail_seq = sp.csr_matrix((d, (row, col)), shape=(num_e * num_r_2, num_e)) # 历史矩阵的压缩存储
#     sp.save_npz('./data/{}/history_seq/train_dev_h_r_history_seq_{}.npz'.format(args.dataset, tim), tail_seq)
#     sp.save_npz('./data/{}/history_seq/train_dev_test_h_r_history_seq_{}.npz'.format(args.dataset, tim), tail_seq)
for idx, tim in tqdm(enumerate(all_times)): # 对于每一个时间戳（从小到大）
    # 取出该时间戳下的所有训练集中的事实四元组 设为(n, 4)
    test_new_data = np.array([[quad[0], quad[1], quad[2], quad[3]] for quad in all_data if quad[3] == tim])
    # get object_entities
    row = test_new_data[:, 0] * num_r_2 + test_new_data[:, 1] # (n,) 一维
    col = test_new_data[:, 2]
    d = np.ones(len(row))
    tail_seq = sp.csr_matrix((d, (row, col)), shape=(num_e * num_r_2, num_e)) # 历史矩阵的压缩存储
    sp.save_npz('./data/{}/history_seq/h_r_history_seq_{}.npz'.format(args.dataset, idx), tail_seq)