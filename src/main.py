import argparse
import os
import sys
import numpy as np
import torch
from tqdm import tqdm
import random
import scipy.sparse as sp
sys.path.append("..")
from rgcn import utils
from rgcn.utils import build_sub_graph, add_inverse_rel
from src.rrgcn import RecurrentRGCN
import torch.nn.modules.rnn
from rgcn.knowledge_graph import _read_triplets_as_list
from collections import defaultdict

import warnings
warnings.filterwarnings(action='ignore')


def test(model, history_list, test_list, num_rels, num_nodes, use_cuda, all_ans_list, model_name, static_graph, static_len, mode):
    '''
    :param model:
    :param history_list: valid传入的是train; test传入的是train+valid (按时间戳划分的事实(内部array, 外层list)：[[[s, r, o], [], ...], [], ...])
    :param test_list: valid传入的是valid; test传入的是test
    :param num_rels: 边的数量，不包括inverse关系
    :param num_nodes:
    :param use_cuda:
    :param all_ans_list:
    :param model_name:
    :param static_graph:
    :param static_len: 静态图的所有时间戳数目
    :param mode:
    :return:
    '''
    ranks_raw, ranks_filter, mrr_raw_list, mrr_filter_list = [], [], [], []

    idx = 0
    if mode == "test":
        # test mode: load parameter form file
        if use_cuda:
            checkpoint = torch.load(model_name, map_location=torch.device(args.gpu))
        else:
            checkpoint = torch.load(model_name, map_location=torch.device('cpu'))
        print("Load Model name: {}. Using best epoch : {}".format(model_name, checkpoint['epoch']))  # use best stat checkpoint
        print("\n"+"-"*10+"start testing"+"-"*10+"\n")
        model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    # 历史时序子图
    history_time_last = len(history_list) - 1 # 最后一个历史时间戳
    test_time_list = [idx for idx in range(history_time_last+1, history_time_last+1+len(test_list))]
    num_rel_2 = num_rels * 2 # 图中的边的数目

    # 静态图结构信息
    if args.add_static_graph:
        all_static_tail_seq = sp.csr_matrix(([], ([], [])), shape=(num_nodes * num_rels * 2, num_nodes))
        for i in range(static_len):
            static_tim_tail_seq = sp.load_npz('../data/{}/history_seq/h_r_history_seq_{}.npz'.format(args.dataset, i))
            all_static_tail_seq = all_static_tail_seq + static_tim_tail_seq

    # test_snap: 对于(valid或者test集中)一个时间戳内的所有事实三元组；time_idx从0开始 一个时间戳一个时间戳地处理
    for time_idx, test_snap in enumerate(tqdm(test_list)): # 对于每一个待测试的时间戳 snapshot_list，按时间戳划分的事实array：[[[s, r, o], [], ...], [], ...]
        # 获取需要进行测试的所有事实query
        triple_with_inverse = add_inverse_rel(test_snap, num_rels)
        seq_idx = triple_with_inverse[:, 0] * num_rel_2 + triple_with_inverse[:, 1]  # 一个时间戳中，对每一个样本（三元组）计算，对应历史矩阵的行
        current_timestamp_idx = test_time_list[time_idx] # 当前时间戳编号（对于整个数据集而言的全局时间戳编号）
        # 计算当前时间戳的历史子图序列
        if time_idx - args.test_history_len < 0:
            input_list = [snap for snap in history_list[time_idx - args.test_history_len:]] + [snap for snap in test_list[0: time_idx]]
        else:
            input_list = [snap for snap in test_list[time_idx - args.test_history_len: time_idx]]

        history_glist = [build_sub_graph(num_nodes, num_rels, g, use_cuda, args.gpu) for g in input_list]
        test_triples_input = torch.LongTensor(test_snap).cuda() if use_cuda else torch.LongTensor(test_snap)
        test_triples_input = test_triples_input.to(args.gpu) # 一个时间戳内的所有事实三元组
        # 获得当前时间戳的历史频次信息和历史mask信息
        history_tail_seq, one_hot_tail_seq = None, None
        if mode == "test":
                    all_tail_seq = sp.csr_matrix(([], ([], [])), shape=(num_nodes * num_rel_2, num_nodes))  # 统计历史事实是否出现
                    for time_idx_inner in range(0, current_timestamp_idx):  # 当前时间戳之前的所有历史时间戳
                        tim_tail_seq = sp.load_npz('../data/{}/history_seq/h_r_history_seq_{}.npz'.format(args.dataset, time_idx_inner))
                        all_tail_seq = all_tail_seq + tim_tail_seq

                    history_tail_seq = torch.Tensor(all_tail_seq[seq_idx].todense())
                    if args.add_static_graph:
                        static_tail_seq = torch.Tensor(all_static_tail_seq[seq_idx].todense())
                        one_hot_tail_seq = static_tail_seq.masked_fill(static_tail_seq != 0, 1)
                    else:
                        one_hot_tail_seq = history_tail_seq.masked_fill(history_tail_seq != 0, 1)

                    if use_cuda:
                        history_tail_seq, one_hot_tail_seq = history_tail_seq.to(args.gpu), one_hot_tail_seq.to(args.gpu)

        # (tensor)all_triples: (batch_size, 3); (tensor)score: (batch_size, num_ents)
        test_triples, final_score = model.predict(history_glist, num_rels, static_graph, history_tail_seq, one_hot_tail_seq, test_triples_input, use_cuda)
        # 每一个时间戳内的事实三元组再按照batch_size进行指标的计算
        mrr_filter_snap, mrr_snap, rank_raw, rank_filter = utils.get_total_rank(test_triples, final_score, all_ans_list[time_idx], eval_bz=1000)

        # used to global statistic
        ranks_raw.append(rank_raw)
        ranks_filter.append(rank_filter)
        # used to show slide results
        mrr_raw_list.append(mrr_snap)
        mrr_filter_list.append(mrr_filter_snap)

        input_list.pop(0)
        input_list.append(test_snap)
        idx += 1
    mrr_raw = utils.stat_ranks(ranks_raw, "raw_ent")
    mrr_filter = utils.stat_ranks(ranks_filter, "filter_ent")
    return mrr_raw, mrr_filter


def run_experiment(args, n_hidden=None, n_layers=None, dropout=None, n_bases=None):
    # load configuration for grid search the best configuration
    if n_hidden:
        args.n_hidden = n_hidden
    if n_layers:
        args.n_layers = n_layers
    if dropout:
        args.dropout = dropout
    if n_bases:
        args.n_bases = n_bases

    # load graph data
    print("loading graph data")
    data = utils.load_data(args.dataset) # data.train, data.valid, data.test: np.array([[s, r, o, time], []...])
    train_list = utils.split_by_time(data.train) # 列表，snapshot_list，按时间戳划分的事实array：[[[s, r, o], [], ...], [], ...]
    valid_list = utils.split_by_time(data.valid)
    test_list = utils.split_by_time(data.test) # snapshot_list，按时间戳划分的事实array：[[[s, r, o], [], ...], [], ...]

    # 静态图信息
    data_list = train_list + valid_list + test_list # 所有的事实三元组array list[[[s, r, o], [], ...], [], ...]
    timestamp_len = len(data_list)

    num_nodes = data.num_nodes # 整个数据集的节点数目
    num_rels = data.num_rels # 整个数据集的关系数目
    num_rel_2 = num_rels * 2 # 图中的关系数目

    # for time-aware filtered evaluation
    all_ans_list_test_time_filter = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes) # data.test: np.array([[s, r, o, time], []...])
    all_ans_list_valid_time_filter = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes)

    model_name = "{}-{}-{}-ly{}-dilate{}-his{}-dp{}|{}|{}|{}-gpu{}"\
        .format(args.dataset, args.encoder, args.decoder, args.n_layers, args.dilate_len, args.train_history_len,
                args.dropout, args.input_dropout, args.hidden_dropout, args.feat_dropout, args.gpu)
    model_state_file = '../models/' + model_name
    print("Sanity Check: stat name : {}".format(model_state_file))
    print("Sanity Check: Is cuda available ? {}".format(torch.cuda.is_available()))

    use_cuda = args.gpu >= 0 and torch.cuda.is_available()

    num_static_rels, num_words, static_triples, static_graph = 0, 0, [], None

    # create stat
    model = RecurrentRGCN(args.decoder,
                          args.encoder,
                        num_nodes,
                        num_rels,
                        num_static_rels,
                        num_words,
                        args.n_hidden,
                        args.opn,
                        sequence_len=args.train_history_len,
                        num_bases=args.n_bases,
                        num_basis=args.n_basis,
                        num_hidden_layers=args.n_layers,
                        dropout=args.dropout,
                        self_loop=args.self_loop,
                        skip_connect=args.skip_connect,
                        layer_norm=args.layer_norm,
                        input_dropout=args.input_dropout,
                        hidden_dropout=args.hidden_dropout,
                        feat_dropout=args.feat_dropout,
                        aggregation=args.aggregation,
                        use_static=args.add_static_graph,
                        entity_prediction=args.entity_prediction,
                        use_cuda=use_cuda,
                        gpu = args.gpu,
                        analysis=args.run_analysis)

    if use_cuda:
        torch.cuda.set_device(args.gpu)
        model.cuda()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    if args.test and os.path.exists(model_state_file):
        mrr_raw, mrr_filter = test(model,
                                   train_list+valid_list,
                                   test_list,
                                   num_rels,
                                   num_nodes,
                                   use_cuda,
                                   all_ans_list_test_time_filter,
                                   model_state_file,
                                   static_graph,
                                   timestamp_len,
                                   "test")
    elif args.test and not os.path.exists(model_state_file):
        print("--------------{} not exist, Change mode to train and generate stat for testing----------------\n".format(model_state_file))
    else:
        print("----------------------------------------start training----------------------------------------\n")
        best_mrr = 0
        for epoch in range(args.n_epochs): # 对于每一个epoch
            model.train()
            # 三个不同的loss
            losses = []
            losses_e = []

            idx = [_ for _ in range(len(train_list))] # train_list: 列表，每一个时间戳内包含的所有事实; idx: 顺序进行的时间戳
            random.shuffle(idx) # 将包含训练集所有时间戳的idx列表打乱

            # 按照时间戳进行训练
            for train_sample_num in tqdm(idx): # 对于每一个时间戳（idx编号是打乱的，但是还对应相应的第几个时间戳）
                if train_sample_num == 0: continue # 第0个时间戳没有历史信息

                # 往下走的必定不是第0个时间戳
                output = train_list[train_sample_num:train_sample_num+1] # 取出当前时间戳下的所有事实array [[s, r, o], [], ...]

                triple_with_inverse = add_inverse_rel(output[0], num_rels)
                seq_idx = triple_with_inverse[:, 0] * num_rel_2 + triple_with_inverse[:, 1] # 一个时间戳中，对每一个样本（三元组）计算，对应历史矩阵的行

                if train_sample_num - args.train_history_len<0: # 当前时间戳的历史深度不够train_history_len
                    input_list = train_list[0: train_sample_num] # 历史信息从第0个时间戳取出来
                else:
                    input_list = train_list[train_sample_num - args.train_history_len: train_sample_num] # 历史信息取前train_history_len个时间戳

                # # 获取历史频次矩阵history_appear_times和历史出现情况矩阵one_hot_tail_seq
                # all_tail_seq = sp.csr_matrix(([], ([], [])), shape=(num_nodes * num_rel_2, num_nodes))  # 统计历史事实是否出现
                # for time_idx in range(0, train_sample_num): # 当前时间戳之前的所有历史时间戳
                #     tim_tail_seq = sp.load_npz('../data/{}/history_seq/h_r_history_seq_{}.npz'.format(args.dataset, time_idx))
                #     all_tail_seq = all_tail_seq + tim_tail_seq
                # history_tail_seq = torch.Tensor(all_tail_seq[seq_idx].todense())
                # one_hot_tail_seq = history_tail_seq.masked_fill(history_tail_seq != 0, 1)
                history_tail_seq, one_hot_tail_seq = None, None
                # if use_cuda:
                #     history_tail_seq, one_hot_tail_seq = history_tail_seq.to(args.gpu), one_hot_tail_seq.to(args.gpu)
                # generate history graph; input_list: [[[s, r, o], [], ...], [], ...] 针对当前时间戳, 前train_history_len个时间戳的历史信息
                history_glist = [build_sub_graph(num_nodes, num_rels, snap, use_cuda, args.gpu) for snap in input_list]
                # 对于当前时间戳下的每一个事实三元组
                output = [torch.from_numpy(_).long().cuda() for _ in output] if use_cuda else [torch.from_numpy(_).long() for _ in output]
                # history_glist：历史子图列表；output[0]：当前时间戳下的所有事实；history_tail_seq：历史频次信息；one_hot_tail_seq：历史mask信息
                loss_e = model.get_loss(history_glist, output[0], static_graph, history_tail_seq, one_hot_tail_seq, use_cuda)
                loss = loss_e

                losses.append(loss.item())
                losses_e.append(loss_e.item())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
                optimizer.step()
                optimizer.zero_grad()

            print("Epoch {:04d} | Ave Loss: {:.4f} | entity:{:.4f} Best MRR {:.4f} | Model {} ".format(epoch, np.mean(losses), np.mean(losses_e), best_mrr, model_name))

            # validation
            if epoch and epoch % args.evaluate_every == 0:
                mrr_raw, mrr_filter = test(model,
                                           train_list, # 列表，snapshot_list，按时间戳划分的事实array：[[[s, r, o], [], ...], [], ...]
                                           valid_list, # [[[s, r, o], [], ...], [], ...]
                                           num_rels,
                                           num_nodes,
                                           use_cuda,
                                           all_ans_list_valid_time_filter,
                                           model_state_file,
                                           static_graph,
                                           timestamp_len,
                                           mode="train")
                
                # entity prediction evalution
                if mrr_raw < best_mrr:
                    if epoch >= args.n_epochs:
                        break
                else:
                    best_mrr = mrr_raw
                    torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)
        mrr_raw, mrr_filter = test(model,
                                   train_list+valid_list,
                                   test_list,
                                   num_rels,
                                   num_nodes,
                                   use_cuda,
                                   all_ans_list_test_time_filter,
                                   model_state_file,
                                   static_graph,
                                   timestamp_len,
                                   mode="test")
    return mrr_raw, mrr_filter


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='REGCN')

    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="batch-size")
    parser.add_argument("-d", "--dataset", type=str, required=True,
                        help="dataset to use")
    parser.add_argument("--test", action='store_true', default=False,
                        help="load stat from dir and directly test")
    parser.add_argument("--run-analysis", action='store_true', default=False,
                        help="print log info")
    parser.add_argument("--run-statistic", action='store_true', default=False,
                        help="statistic the result")
    parser.add_argument("--multi-step", action='store_true', default=False,
                        help="do multi-steps inference without ground truth")
    parser.add_argument("--topk", type=int, default=10,
                        help="choose top k entities as results when do multi-steps without ground truth")
    parser.add_argument("--add-static-graph",  action='store_true', default=False,
                        help="use the info of static graph")
    parser.add_argument("--add-rel-word", action='store_true', default=False,
                        help="use words in relaitons")
    parser.add_argument("--relation-evaluation", action='store_true', default=False,
                        help="save model accordding to the relation evalution")

    # configuration for encoder RGCN stat
    parser.add_argument("--encoder", type=str, default="uvrgcn",
                        help="method of encoder")
    parser.add_argument("--aggregation", type=str, default="none",
                        help="method of aggregation")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="dropout probability")
    parser.add_argument("--skip-connect", action='store_true', default=False,
                        help="whether to use skip connect in a RGCN Unit")
    parser.add_argument("--n-hidden", type=int, default=200,
                        help="number of hidden units")
    parser.add_argument("--opn", type=str, default="sub",
                        help="opn of compgcn")

    parser.add_argument("--n-bases", type=int, default=100,
                        help="number of weight blocks for each relation")
    parser.add_argument("--n-basis", type=int, default=100,
                        help="number of basis vector for compgcn")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="number of propagation rounds")
    parser.add_argument("--self-loop", action='store_true', default=True,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--layer-norm", action='store_true', default=False,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--entity-prediction", action='store_true', default=False,
                        help="add entity prediction loss")
    parser.add_argument("--split_by_relation", action='store_true', default=False,
                        help="do relation prediction")

    # configuration for stat training
    parser.add_argument("--n-epochs", type=int, default=500,
                        help="number of minimum training epochs on each time step")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--grad-norm", type=float, default=1.0,
                        help="norm to clip gradient to")

    # configuration for evaluating
    parser.add_argument("--evaluate-every", type=int, default=20,
                        help="perform evaluation every n epochs")

    # configuration for decoder
    parser.add_argument("--decoder", type=str, default="convtranse",
                        help="method of decoder")
    parser.add_argument("--input-dropout", type=float, default=0.2,
                        help="input dropout for decoder ")
    parser.add_argument("--hidden-dropout", type=float, default=0.2,
                        help="hidden dropout for decoder")
    parser.add_argument("--feat-dropout", type=float, default=0.2,
                        help="feat dropout for decoder")

    # configuration for sequences stat
    parser.add_argument("--train-history-len", type=int, default=10,
                        help="history length")
    parser.add_argument("--test-history-len", type=int, default=20,
                        help="history length for test")
    parser.add_argument("--dilate-len", type=int, default=1,
                        help="dilate history graph")

    # configuration for optimal parameters
    parser.add_argument("-tune", "--tune", type=str, default="n_hidden,n_layers,dropout,n_bases",
                        help="stat to use")
    parser.add_argument("--num-k", type=int, default=500,
                        help="number of triples generated")


    args = parser.parse_args()
    print(args)
    run_experiment(args)
    sys.exit()



