#coding: utf-8
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from numpy_artm import *
import numpy as np
from scipy.sparse import csr_matrix
import time
import pickle

data = np.load('/home/tylorn/data.npy')
indices = np.load('/home/tylorn/indices.npy')
indptr = np.load('/home/tylorn/indptr.npy')

n_dw_matrix = csr_matrix((data, indices, indptr))
n_dw_matrix.eliminate_zeros()


def custom_reg(n_tw, n_dt):
    return 0., 0.


for seed in xrange(777, 779):
    T = 100
    D, W = n_dw_matrix.shape
    np.random.seed(seed)

    phi_matrix = np.random.uniform(size=(T, W)).astype(np.float64)
    phi_matrix /= np.sum(phi_matrix, axis=1)[:, np.newaxis]

    theta_matrix = np.transpose(phi_matrix)
    theta_matrix /= np.sum(theta_matrix, axis=1)[:, np.newaxis]

    regularization_list = np.zeros(200, dtype=object)
    regularization_list[:] = custom_reg
    
    start = time.time()
    def callback(it, phi, theta):
        global start
        print it, time.time() - start
        start = time.time()
        print '\tsparsity', 1. * np.sum(phi < 1e-20) / np.sum(phi >= 0)
        print '\ttheta_sparsity', 1. * np.sum(theta < 1e-20) / np.sum(theta >= 0)

    phi, theta, n_tw, n_dt = em_optimization(
        n_dw_matrix=n_dw_matrix, 
        phi_matrix=phi_matrix,
        theta_matrix=theta_matrix,
        regularization_list=regularization_list,
        iters_count=60,
        iteration_callback=callback,
        params={'return_counters': True}
    )

    _train_perplexity = artm_calc_perplexity_factory(n_dw_matrix) 
    print 'train_perplexity', _train_perplexity(phi, theta)
    print 'topic_correlation', artm_calc_topic_correlation(phi)
    print 'sparsity', 1. * np.sum(phi < 1e-20) / np.sum(phi >= 0)
    print 'theta_sparsity', 1. * np.sum(theta < 1e-20) / np.sum(theta >= 0)
    print 'kernel_avg_size', np.mean(artm_get_kernels_sizes(phi))
    print 'kernel_avg_jacard', artm_get_avg_pairwise_kernels_jacards(phi)

    print 'top10_avg_jacard', artm_get_avg_top_words_jacards(phi, 10)
    print 'top50_avg_jacard', artm_get_avg_top_words_jacards(phi, 50)
    print 'top100_avg_jacard', artm_get_avg_top_words_jacards(phi, 100)
    print 'top200_avg_jacard', artm_get_avg_top_words_jacards(phi, 200)

    with open('plsa_origin/exp_seed_{}.pkl'.format(seed), 'w') as f:
        pickle.dump({
            'phi': phi, 
            'theta': theta, 
            'n_tw': n_tw, 
            'n_dt': n_dt
        }, f)

