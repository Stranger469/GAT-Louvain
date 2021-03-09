#coding=utf-8
from __future__ import division
from __future__ import print_function

import os
import argparse
import tensorflow as tf
import tensorflow.compat.v1 as v1
import numpy as np
import time

from .utils import *
from .create_adjacency import *
from .minibatch import MinibatchIterator
from .model import DGRec

seed = 123
np.random.seed(seed)
v1.set_random_seed(seed)

def evaluate(sess, model, minibatch, val_or_test='val'):
    epoch_val_cost = []
    epoch_val_recall = []
    epoch_val_ndcg = []
    epoch_val_point = []
    while not minibatch.end_val(val_or_test):
        feed_dict = minibatch.next_val_minibatch_feed_dict(val_or_test)
        outs = sess.run([model.loss, model.sum_recall, model.sum_ndcg, model.point_count], feed_dict=feed_dict)
        epoch_val_cost.append(outs[0])
        epoch_val_recall.append(outs[1])
        epoch_val_ndcg.append(outs[2])
        epoch_val_point.append(outs[3])
    return np.mean(epoch_val_cost), np.sum(epoch_val_recall) / np.sum(epoch_val_point), np.sum(epoch_val_ndcg) / np.sum(epoch_val_point)

def construct_placeholders(args):
    # Define placeholders
    placeholders = {
        'input_x': v1.placeholder(tf.int32, shape=(args.batch_size, args.max_length), name='input_session'),
        'input_y': v1.placeholder(tf.int32, shape=(args.batch_size, args.max_length), name='output_session'),
        'mask_y': v1.placeholder(tf.float32, shape=(args.batch_size, args.max_length), name='mask_x'),
        'support_nodes_layer1': v1.placeholder(tf.int32, shape=(args.batch_size*args.samples_1*args.samples_2), name='support_nodes_layer1'),
        'support_nodes_layer2': v1.placeholder(tf.int32, shape=(args.batch_size*args.samples_2), name='support_nodes_layer2'),
        'support_sessions_layer1': v1.placeholder(tf.int32, shape=(args.batch_size*args.samples_1*args.samples_2,\
                                    args.max_length), name='support_sessions_layer1'),
        'support_sessions_layer2': v1.placeholder(tf.int32, shape=(args.batch_size*args.samples_2,\
                                    args.max_length), name='support_sessions_layer2'),
        'support_lengths_layer1': v1.placeholder(tf.int32, shape=(args.batch_size*args.samples_1*args.samples_2), 
                                    name='support_lengths_layer1'),
        'support_lengths_layer2': v1.placeholder(tf.int32, shape=(args.batch_size*args.samples_2), 
                                    name='support_lengths_layer2'),
    }
    return placeholders

def train(args, minibatch):
    v1.get_logger().setLevel('ERROR')

    args.num_items = len(item_id_map) + 1
    args.num_users = len(user_id_map)
    # placeholders = construct_placeholders(args)
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    print(args.ckpt_dir, flush=True)
    ckpt_path = os.path.join(args.ckpt_dir, 'model.ckpt')
    print(ckpt_path, flush=True)

    print('minibatch initialized', flush=True)
    dgrec = DGRec(args, minibatch.sizes, placeholders)
    
    config = v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = v1.Session(config=config)
    sess.run(v1.global_variables_initializer())
    saver = v1.train.Saver(v1.global_variables(), max_to_keep=3)

    total_steps = 0
    avg_time = 0.

    patience = 10
    inc = 0
    early_stopping = False

    highest_val_recall = -1.0
    start_time = time.time()
    for epoch in range(args.epochs):
        minibatch.shuffle()

        iter_cn = 0
        print('Epoch: %04d' % (epoch + 1))
        epoch_val_cost = []
        epoch_val_recall = []
        epoch_val_ndcg = []
        epoch_train_cost = []
        epoch_train_recall = []
        epoch_train_ndcg = []
        epoch_train_point = []
        
        while not minibatch.end() and not early_stopping:
            # print('start')
            t = time.time()
            feed_dict = minibatch.next_train_minibatch_feed_dict()
            # print('point1')
            outs = sess.run([dgrec.opt_op, dgrec.loss, dgrec.sum_recall, dgrec.sum_ndcg, dgrec.point_count], feed_dict=feed_dict)
            # print('point2')
            train_cost = outs[1]
            epoch_train_cost.append(train_cost)
            epoch_train_recall.append(outs[2])
            epoch_train_ndcg.append(outs[3])
            epoch_train_point.append(outs[4])
            # Print results
            avg_time = (avg_time * total_steps + time.time() - t) / (total_steps + 1)

            if iter_cn % args.val_every == 0:
                ret = evaluate(sess, dgrec, minibatch)
                # print('point3')
                epoch_val_cost.append(ret[0])
                epoch_val_recall.append(ret[1])
                epoch_val_ndcg.append(ret[2])
                if ret[1] >= highest_val_recall:
                    saver.save(sess, ckpt_path, global_step=total_steps)
                    highest_val_recall = ret[1]
                    inc = 0
                    print("Iter:", '%d' % iter_cn, 
                          "val_loss=", "{:.5f}".format(epoch_val_cost[-1]),
                          "val_recall@20=", "{:.5f}".format(epoch_val_recall[-1]),
                          "val_ndcg=", "{:.5f}".format(epoch_val_ndcg[-1]),
                          "dump model!", 
                          flush=True)
                else:
                    inc += 1
                if inc >= patience:
                    early_stopping = True
                    break

            if total_steps % args.print_every == 0:
                print("Iter:", '%d' % iter_cn, 
                      "train_loss=", "{:.5f}".format(np.mean(epoch_train_cost)),
                      "train_recall@20=", "{:.5f}".format(np.sum(epoch_train_recall)/np.sum(epoch_train_point)),
                      "train_ndcg=", "{:.5f}".format(np.sum(epoch_train_ndcg)/np.sum(epoch_train_point)),
                      "val_loss=", "{:.5f}".format(epoch_val_cost[-1]),
                      "val_recall@20=", "{:.5f}".format(epoch_val_recall[-1]),
                      "val_ndcg=", "{:.5f}".format(epoch_val_ndcg[-1]),
                      "time=", "{:.5f}s".format(avg_time), flush=True)
            total_steps += 1
            iter_cn += 1
            # print('end')
        if early_stopping:
            print('Early stop at epoch: {}, total training steps: {}'.format(epoch, total_steps), flush=True)
            break
    
    end_time = time.time() 
    print('-----------{} seconds per batch iteration-------------'.format((end_time - start_time) / total_steps), flush=True)
    print('Parameter settings: {}'.format(args.ckpt_dir), flush=True)
    print('Optimization finished!\tStart testing...', flush=True)
    ret = evaluate(sess, dgrec, minibatch, 'test')
    print('Test results:',
            '\tLoss:{}'.format(ret[0]),
            '\tRecall@20:{}'.format(ret[1]),
            '\tNDCG:{}'.format(ret[2]), flush=True)
    
class Args():
    training = True
    global_only = False
    local_only = False
    epochs = 20
    aggregator_type='attn'
    act='relu'
    batch_size = 200
    max_degree = 50
    num_users = -1
    num_items = 100
    concat=False
    learning_rate=0.001
    hidden_size = 100
    embedding_size = 50
    emb_user = 50
    max_length=20
    samples_1=10
    samples_2=5
    dim1 = 100
    dim2 = 100
    model_size = 'small'
    dropout = 0.
    weight_decay = 0.
    print_every = 100
    val_every = 500
    ckpt_dir = 'save/'

def parseArgs():
    args = Args()
    parser = argparse.ArgumentParser(description='DGRec args')
    parser.add_argument('--batch', default=200, type=int)
    parser.add_argument('--model', default='attn', type=str)
    parser.add_argument('--act', default='relu', type=str)
    parser.add_argument('--degree', default=50, type=int)
    parser.add_argument('--lr', default=0.002, type=float)
    parser.add_argument('--hidden', default=100, type=int)
    parser.add_argument('--embi', default=50, type=int)
    parser.add_argument('--embu', default=50, type=int)
    parser.add_argument('--samples1', default=10, type=int)
    parser.add_argument('--samples2', default=5, type=int)
    parser.add_argument('--dim1', default=100, type=int)
    parser.add_argument('--dim2', default=100, type=int)
    parser.add_argument('--dropout', default=0., type=float)
    parser.add_argument('--l2', default=0., type=float)
    parser.add_argument('--decay_steps', default=400, type=int)
    parser.add_argument('--decay_rate', default=0.98, type=float)
    parser.add_argument('--local', default=0, type=int)
    parser.add_argument('--glb', default=0, type=int)
    new_args = parser.parse_args()

    args.batch_size = new_args.batch
    args.aggregator_type = new_args.model
    args.act = new_args.act
    args.max_degree = new_args.degree
    args.learning_rate = new_args.lr
    args.hidden_size = new_args.hidden
    args.embedding_size = new_args.embi
    args.emb_user = new_args.embu
    args.samples_1 = new_args.samples1
    args.samples_2 = new_args.samples2
    args.dim1 = new_args.dim1
    args.dim2 = new_args.dim2
    args.dropout = new_args.dropout
    args.weight_decay = new_args.l2
    args.decay_steps = new_args.decay_steps
    args.decay_rate = new_args.decay_rate
    args.local_only = new_args.local
    args.global_only = new_args.glb
    args.ckpt_dir = args.ckpt_dir + 'dgrec_batch{}'.format(args.batch_size)
    # args.ckpt_dir = args.ckpt_dir + '_model{}'.format(args.aggregator_type)
    # args.ckpt_dir = args.ckpt_dir + '_act{}'.format(args.act)
    # args.ckpt_dir = args.ckpt_dir + '_maxdegree{}'.format(args.max_degree)
    # args.ckpt_dir = args.ckpt_dir + '_lr{}'.format(args.learning_rate)
    # args.ckpt_dir = args.ckpt_dir + '_hidden{}'.format(args.hidden_size)
    args.ckpt_dir = args.ckpt_dir + '_embi{}'.format(args.embedding_size)
    args.ckpt_dir = args.ckpt_dir + '_embu{}'.format(args.emb_user)
    args.ckpt_dir = args.ckpt_dir + '_samples1st{}'.format(args.samples_1)
    args.ckpt_dir = args.ckpt_dir + '_samples2nd{}'.format(args.samples_2)
    # args.ckpt_dir = args.ckpt_dir + '_dim1st{}'.format(args.dim1)
    # args.ckpt_dir = args.ckpt_dir + '_dim2nd{}'.format(args.dim2)
    # args.ckpt_dir = args.ckpt_dir + '_dropout{}'.format(args.dropout)
    # args.ckpt_dir = args.ckpt_dir + '_l2reg{}'.format(args.weight_decay)
    # args.ckpt_dir = args.ckpt_dir + '_decaysteps{}'.format(args.decay_steps)
    # args.ckpt_dir = args.ckpt_dir + '_decayrate{}'.format(args.decay_rate)
    # args.ckpt_dir = args.ckpt_dir + '_global{}'.format(new_args.glb)
    # args.ckpt_dir = args.ckpt_dir + '_local{}'.format(new_args.local)
    return args
 

if __name__ == '__main__':
    # v1.app.run()
    v1.disable_eager_execution()
    v1.disable_v2_behavior()
    v1.get_logger().setLevel('ERROR')

    args = parseArgs()
    
    print('Loading training data..', flush=True)
    data = load_data('data\\data\\yelp\\')
    print("Training data loaded!", flush=True)
    adj_info = data[0]
    latest_per_user_by_time = data[1]
    user_id_map = data[2]
    item_id_map = data[3]
    train_df = data[4]
    valid_df = data[5]
    test_df = data[6]

    args.num_items = len(item_id_map) + 1
    args.num_users = len(user_id_map)

    print('start minibatch construction')
    # TODO: 整合得到的adj
    test_adj, test_deg = construct_adj_multiProcess(
        data=pd.concat([train_df, valid_df, test_df]),
        num_nodes=len(user_id_map),
        max_degree=50,
        adj_info=adj_info,
        num_process=10)
    train_adj, train_deg = construct_adj_multiProcess(
        data=train_df,
        num_nodes=len(user_id_map),
        max_degree=50,
        adj_info=adj_info,
        num_process=10)
    placeholders = construct_placeholders(args)

    minibatch = MinibatchIterator(adj_info,
                latest_per_user_by_time,
                [train_df, valid_df, test_df],
                placeholders,
                batch_size=args.batch_size,
                max_degree=args.max_degree,
                num_nodes=len(user_id_map),
                test_adj = test_adj,
                test_deg = test_deg,
                train_adj = train_adj,
                train_deg = train_deg,
                max_length=args.max_length,
                samples_1_2=[args.samples_1, args.samples_2])

    train(args, minibatch)
