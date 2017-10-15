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

    train_n_dw_matrix, test_n_dw_matrix, _, _, _ = prepare_dataset(dataset, train_test_split=0.8)

    D, W = train_n_dw_matrix.shape

    train_perplexities = []
    test_perplexities = []
    sparsities = []
    theta_sparsities = []
    uniqueness_measures = []
    normalized_uniqueness_measures = []

    for seed in xrange(samples):
        print seed
        train_perplexity = []
        test_perplexity = []
        sparsity = []
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
        _test_perplexity = artm_calc_perplexity_factory(test_n_dw_matrix) 

        def callback(it, phi, theta):
            if it % 5 == 0:
                train_perplexity.append(_train_perplexity(phi, theta))
                test_perplexity.append(_test_perplexity(phi, theta))
                sparsity.append(1. * np.sum(phi < 1e-20) / np.sum(phi >= 0))
                theta_sparsity.append(1. * np.sum(theta < 0.01) / np.sum(theta >= 0))
                ums, nums = artm_calc_phi_uniqueness_measures(phi)
                uniqueness_measure.append(ums)
                normalized_uniqueness_measure.append(nums)


        phi, theta = optimization_method(
            n_dw_matrix=train_n_dw_matrix, 
            phi_matrix=phi_matrix,
            theta_matrix=theta_matrix,
            regularization_list=regularization_list,
            iters_count=iters_count,
            iteration_callback=callback,
            params=params
        )

        train_perplexities.append(train_perplexity)
        test_perplexities.append(test_perplexity)
        sparsities.append(sparsity)
        theta_sparsities.append(theta_sparsity)
        uniqueness_measures.append(uniqueness_measure)
        normalized_uniqueness_measures.append(normalized_uniqueness_measure)


    with open(output_path, 'w') as f:
        pickle.dump({
            'train_perplexities': train_perplexities,
            'test_perplexities': test_perplexities,
            'sparsities': sparsities,
            'theta_sparsities': theta_sparsities,
            'uniqueness_measures': uniqueness_measures,
            'normalized_uniqueness_measures': normalized_uniqueness_measures,
        }, f)

if __name__ == '__main__':
    args_list = []
    for t in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
        args_list.append(
            (
                em_optimization, 
                t, 100, 10,
                -0.1, 0.,
                {},
                'iter_exp/iter_exp_20news_{}t_base_-0.1_0.pkl'.format(t)
            )
        )

    Pool(processes=5).map(perform_experiment, args_list)
