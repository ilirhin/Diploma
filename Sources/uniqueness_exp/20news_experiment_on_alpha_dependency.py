#coding: utf-8
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from numpy_artm import *
from multiprocessing import Pool, Manager
import pickle


def perform_experiment((
    optimization_method,
    T, iters_count, samples,
    phi_alpha, theta_alpha, 
    params,
    output_path
)):
    dataset = fetch_20newsgroups(
        subset='all',
        categories=[
            'sci.crypt',
            'sci.electronics',
            'sci.med',
            'sci.space'
        ],
        remove=('headers', 'footers', 'quotes')
    )

    train_n_dw_matrix, _, _, _ = prepare_dataset(dataset)

    D, W = train_n_dw_matrix.shape

    train_perplexities = []
    sparsities = []
    theta_sparsities = []
    uniqueness_measures = []
    normalized_uniqueness_measures = []

    for seed in xrange(samples):
        print seed
        train_perplexity = []
        sparsitiy = []
        theta_sparsity = []
        uniqueness_measure = []
        normalized_uniqueness_measure = []

        np.random.seed(seed)

        phi_matrix = np.random.uniform(size=(T, W)).astype(np.float64)
        phi_matrix /= np.sum(phi_matrix, axis=1)[:, np.newaxis]

        theta_matrix = np.ones(shape=(D, T)).astype(np.float64)
        theta_matrix /= np.sum(theta_matrix, axis=1)[:, np.newaxis]

        regularization_list = np.zeros(200, dtype=object)
        regularization_list[:] = create_reg_lda(phi_alpha, theta_alpha)

        _train_perplexity = artm_calc_perplexity_factory(train_n_dw_matrix) 

        phi, theta = optimization_method(
            n_dw_matrix=train_n_dw_matrix, 
            phi_matrix=phi_matrix,
            theta_matrix=theta_matrix,
            regularization_list=regularization_list,
            iters_count=iters_count,
            iteration_callback=None,
            params=params
        )

        train_perplexities.append(_train_perplexity(phi, theta))
        sparsities.append(1. * np.sum(phi < 1e-20) / np.sum(phi >= 0))
        theta_sparsities.append(1. * np.sum(theta < 0.01) / np.sum(theta >= 0))
        ums, nums = artm_calc_phi_uniqueness_measures(phi)
        uniqueness_measures.append(ums)
        normalized_uniqueness_measures.append(nums)
        print np.min(ums), np.min(nums)
        

    with open(output_path, 'w') as f:
        pickle.dump({
            'train_perplexities': train_perplexities,
            'sparsities': sparsities,
            'theta_sparsities': theta_sparsities,
            'uniqueness_measures': uniqueness_measures,
            'normalized_uniqueness_measures': normalized_uniqueness_measures,
        }, f)

if __name__ == '__main__':
    alpha_values = [
        - 10 ** (-i)
        for i in xrange(30)
    ]
    args_list = []
    for alpha in alpha_values:
        for beta in [-0.1, 0, 0.1]:
            args_list.append((
                em_optimization, 
                10, 100, 200,
                alpha, beta,
                {},
                'alpha_exp/alpha_exp_20news_10t_base_{}_{}.pkl'.format(alpha, beta)
            ))

    Pool(processes=3).map(perform_experiment, args_list)
