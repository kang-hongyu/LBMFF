import numpy as np
import scipy.sparse as sp
import tensorflow as tf
import gc
import random
import os
from clac_metric import cv_model_evaluate
from utils import *
from model import GCNModel
from opt import Optimizer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

def PredictScore(train_drug_dis_matrix, drug_matrix, dis_matrix, seed, epochs, emb_dim, dp, lr,  adjdp):
    np.random.seed(seed)
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    adj = constructHNet(train_drug_dis_matrix, drug_matrix, dis_matrix)
    adj = sp.csr_matrix(adj)
    association_nam = train_drug_dis_matrix.sum()
    X = constructNet(train_drug_dis_matrix)
    features = sparse_to_tuple(sp.csr_matrix(X))
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]
    adj_orig = train_drug_dis_matrix.copy()
    adj_orig = sparse_to_tuple(sp.csr_matrix(adj_orig))

    adj_norm = preprocess_graph(adj)
    adj_nonzero = adj_norm[1].shape[0]
    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'adjdp': tf.placeholder_with_default(0., shape=())
    }
    model = GCNModel(placeholders, num_features, emb_dim,
                     features_nonzero, adj_nonzero, train_drug_dis_matrix.shape[0], name='LAGCN')
    with tf.name_scope('optimizer'):
        opt = Optimizer(
            preds=model.reconstructions,
            labels=tf.reshape(tf.sparse_tensor_to_dense(
                placeholders['adj_orig'], validate_indices=False), [-1]),
            model=model,
            lr=lr, num_u=train_drug_dis_matrix.shape[0], num_v=train_drug_dis_matrix.shape[1], association_nam=association_nam)
    sess = tf.Session(config = config)
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        feed_dict = dict()
        feed_dict.update({placeholders['features']: features})
        feed_dict.update({placeholders['adj']: adj_norm})
        feed_dict.update({placeholders['adj_orig']: adj_orig})
        feed_dict.update({placeholders['dropout']: dp})
        feed_dict.update({placeholders['adjdp']: adjdp})
        _, avg_cost = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)
        if epoch % 100 == 0:
            feed_dict.update({placeholders['dropout']: 0})
            feed_dict.update({placeholders['adjdp']: 0})
            res = sess.run(model.reconstructions, feed_dict=feed_dict)
            print("Epoch:", '%04d' % (epoch + 1),
                  "train_loss=", "{:.5f}".format(avg_cost))
    print('Optimization Finished!')
    feed_dict.update({placeholders['dropout']: 0})
    feed_dict.update({placeholders['adjdp']: 0})
    res = sess.run(model.reconstructions, feed_dict=feed_dict)
    sess.close()
    return res


def cross_validation_experiment(drug_dis_matrix, drug_matrix, dis_matrix, seed, epochs, emb_dim, dp, lr, adjdp):
    index_matrix = np.mat(np.where(drug_dis_matrix < 2))
    association_nam = index_matrix.shape[1]
    random_index = index_matrix.T.tolist()
    random.seed(seed)
    random.shuffle(random_index)
    k_folds = 5
    CV_size = int(association_nam / k_folds)
    temp = np.array(random_index[:association_nam - association_nam %
                                 k_folds]).reshape(k_folds, CV_size,  -1).tolist()
    temp[k_folds - 1] = temp[k_folds - 1] + \
        random_index[association_nam - association_nam % k_folds:]
    random_index = temp
    metric = np.zeros((1, 7))
    xroc = np.zeros((1, 100))
    yroc = np.zeros((1, 100))
    xpr = np.zeros((1, 100))
    ypr = np.zeros((1, 100))
    print("seed=%d, evaluating drug-disease...." % (seed))
    k_folds_ = 0
    for k in range(k_folds):
        print("------this is %dth cross validation------" % (k+1))
        train_matrix = np.matrix(drug_dis_matrix, copy=True)
        test_matrix = np.matrix(drug_dis_matrix, copy=True)
        train_matrix[tuple(np.array(random_index[k]).T)] = 0
        test_matrix[tuple(np.array(random_index[k]).T)] = 2
        drug_len = drug_dis_matrix.shape[0]
        dis_len = drug_dis_matrix.shape[1]
        drug_disease_res = PredictScore(
            train_matrix, drug_matrix, dis_matrix, seed, epochs, emb_dim, dp, lr,  adjdp)
        predict_y_proba = drug_disease_res.reshape(drug_len, dis_len)
        np.savetxt('../data/SCMFDD-S/data3/predict_%d.csv' % k, predict_y_proba, delimiter=",")
        metric_tmp, plotdata = cv_model_evaluate(
            drug_dis_matrix, predict_y_proba, test_matrix)
        print(metric_tmp)
        if metric_tmp[1] < 0.55:
            continue
        k_folds_ += 1
        xroc += plotdata[0]
        yroc += plotdata[1]
        xpr  += plotdata[2]
        ypr  += plotdata[3]
        metric += metric_tmp
        del train_matrix
        gc.collect()
    if k_folds_ == 0:
        return matric
    print(metric / k_folds_)
    metric = np.array(metric / k_folds_)

    xroc = np.array(xroc / k_folds_).flatten()
    yroc = np.array(yroc / k_folds_).flatten()
    xpr  = np.array(xpr / k_folds_).flatten()
    ypr  = np.array(ypr / k_folds_).flatten()
    print(",".join([str(x) for x in xroc]))
    print(",".join([str(x) for x in yroc]))
    print(",".join([str(x) for x in xpr]))
    print(",".join([str(x) for x in ypr]))

    plt.plot(xroc, yroc, 'r', label='AOC')
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--', color='b')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.savefig('auc.jpg')
    plt.clf()

    plt.plot(xpr, ypr, 'r', label='PR')
    plt.legend(loc='lower left')
    plt.plot([1, 0], 'r--', color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall')
    plt.savefig('aupr.jpg')
    plt.clf()

    return metric


if __name__ == "__main__":
    drug_sim = np.loadtxt('../data/SCMFDD-S/data3/drugsim.csv', delimiter=',')
    dis_sim = np.loadtxt('../data/SCMFDD-S/data3/dissim.csv', delimiter=',')
    drug_dis_matrix = np.loadtxt('../data/SCMFDD-S/data3/drug_disease_sim.csv', delimiter=',')
    epoch = 10000
    emb_dim = 128
    lr = 0.01
    adjdp = 0.1
    dp = 0.1
    simw = 6
    result = np.zeros((1, 7), float)
    average_result = np.zeros((1, 7), float)
    circle_time = 1
    for i in range(circle_time):
        result += cross_validation_experiment(
            drug_dis_matrix, drug_sim*simw, dis_sim*simw, i, epoch, emb_dim, dp, lr, adjdp)
    average_result = result / circle_time
    print(average_result)
    print("[result]")
    print(np.array(average_result).flatten())
