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
            'rec.autos',
            'rec.motorcycles',
            'rec.sport.baseball',
            'rec.sport.hockey',
            'sci.crypt',
            'sci.electronics',
            'sci.med',
            'sci.space'
        ],
        remove=('headers', 'footers', 'quotes')
    )

    train_n_dw_matrix, test_n_dw_matrix, _, _, doc_targets, doc_occurences, doc_cooccurences = prepare_dataset(dataset, calc_cooccurences=True, train_test_split=0.8)

    D, W = train_n_dw_matrix.shape

    train_perplexities = []
    test_perplexities = []
    sparsities = []
    theta_sparsities = []
    topic_correlations = []
    avg_top5_pmis = []
    avg_top10_pmis = []
    avg_top20_pmis = []
    avg_top30_pmis = []
    kernel_avg_sizes = []
    kernel_avg_jacards = []

    for seed in xrange(samples):
        print seed
        train_perplexity = []
        test_perplexity = []
        sparsity = []
        theta_sparsity = []
        topic_correlation = []
        avg_top5_pmi = []
        avg_top10_pmi = []
        avg_top20_pmi = []
        avg_top30_pmi = []
        kernel_avg_size = []
        kernel_avg_jacard = []

        np.random.seed(seed)

        phi_matrix = np.random.uniform(size=(T, W)).astype(np.float64) + 0.1
        phi_matrix /= np.sum(phi_matrix, axis=1)[:, np.newaxis]

        theta_matrix = np.ones(shape=(D, T)).astype(np.float64)
        theta_matrix /= np.sum(theta_matrix, axis=1)[:, np.newaxis]

        regularization_list = np.zeros(200, dtype=object)
        regularization_list[:] = create_reg_lda(phi_alpha, theta_alpha)

        _train_perplexity = artm_calc_perplexity_factory(train_n_dw_matrix) 
        _test_perplexity = artm_calc_perplexity_factory(test_n_dw_matrix)
        _top5_pmi = artm_calc_pmi_top_factory(doc_occurences, doc_cooccurences, train_n_dw_matrix.shape[0], 5)
        _top10_pmi = artm_calc_pmi_top_factory(doc_occurences, doc_cooccurences, train_n_dw_matrix.shape[0], 10)
        _top20_pmi = artm_calc_pmi_top_factory(doc_occurences, doc_cooccurences, train_n_dw_matrix.shape[0], 20)
        _top30_pmi = artm_calc_pmi_top_factory(doc_occurences, doc_cooccurences, train_n_dw_matrix.shape[0], 30)

        def callback(it, phi, theta):
            train_perplexity.append(_train_perplexity(phi, theta))
            test_perplexity.append(_test_perplexity(phi, theta))
            topic_correlation.append(artm_calc_topic_correlation(phi))
            sparsity.append(1. * np.sum(phi < 1e-20) / np.sum(phi >= 0))
            theta_sparsity.append(1. * np.sum(theta < 0.01) / np.sum(theta >= 0))
            avg_top5_pmi.append(_top5_pmi(phi))
            avg_top10_pmi.append(_top10_pmi(phi))
            avg_top20_pmi.append(_top20_pmi(phi))
            avg_top30_pmi.append(_top30_pmi(phi))
            kernel_avg_size.append(np.mean(artm_get_kernels_sizes(phi)))
            kernel_avg_jacard.append(artm_get_avg_pairwise_kernels_jacards(phi))


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
        topic_correlations.append(topic_correlation)
        avg_top5_pmis.append(avg_top5_pmi)
        avg_top10_pmis.append(avg_top10_pmi)
        avg_top20_pmis.append(avg_top20_pmi)
        avg_top30_pmis.append(avg_top30_pmi)
        kernel_avg_sizes.append(kernel_avg_size)
        kernel_avg_jacards.append(kernel_avg_jacard)

    with open(output_path, 'w') as f:
        pickle.dump({
            'train_perplexities': train_perplexities,
            'test_perplexities': test_perplexities,
            'sparsities': sparsities,
            'theta_sparsities': theta_sparsities,
            'topic_correlations': topic_correlations,
            'avg_top5_pmis': avg_top5_pmis,
            'avg_top10_pmis': avg_top10_pmis,
            'avg_top20_pmis': avg_top20_pmis,
            'avg_top30_pmis': avg_top30_pmis,
            'kernel_avg_sizes': kernel_avg_sizes,
            'kernel_avg_jacards': kernel_avg_jacards
        }, f)

if __name__ == '__main__':
    args_list = [
        (
            em_optimization, 
            10, 100, 100,
            0., 0.,
            {},
            '20news_10t_base_0_0.pkl'
        ),
        (
            artm_thetaless_em_optimization, 
            10, 100, 100,
            0., 0.,
            {'use_B_cheat': False},
            '20news_10t_artm_0_0.pkl'
        ),
        (
            artm_thetaless_em_optimization, 
            10, 100, 100,
            0., 0.,
            {'use_B_cheat': True},
            '20news_10t_artm_0_0_cheat.pkl'
        ),
        (
            em_optimization, 
            10, 100, 100,
            -0.1, 0.,
            {},
            '20news_10t_base_-0.1_0.pkl'
        ),
        (
            artm_thetaless_em_optimization, 
            10, 100, 100,
            -0.1, 0.,
            {'use_B_cheat': False},
            '20news_10t_artm_-0.1_0.pkl'
        ),
        (
            artm_thetaless_em_optimization, 
            10, 100, 100,
            -0.1, 0.,
            {'use_B_cheat': True},
            '20news_10t_artm_-0.1_0_cheat.pkl'
        ),
        (
            em_optimization, 
            10, 100, 100,
            0.1, 0.,
            {},
            '20news_10t_base_+0.1_0.pkl'
        ),
        (
            artm_thetaless_em_optimization, 
            10, 100, 100,
            0.1, 0.,
            {'use_B_cheat': False},
            '20news_10t_artm_+0.1_0.pkl'
        ),
        (
            artm_thetaless_em_optimization, 
            10, 100, 100,
            0.1, 0.,
            {'use_B_cheat': True},
            '20news_10t_artm_+0.1_0_cheat.pkl'
        ),
        (
            em_optimization, 
            10, 100, 100,
            0., -0.1,
            {},
            '20news_10t_base_0_-0.1.pkl'
        ),
        (
            artm_thetaless_em_optimization, 
            10, 100, 100,
            0., -0.1,
            {'use_B_cheat': False},
            '20news_10t_artm_0_-0.1.pkl'
        ),
        (
            artm_thetaless_em_optimization, 
            10, 100, 100,
            0., -0.1,
            {'use_B_cheat': True},
            '20news_10t_artm_0_-0.1_cheat.pkl'
        ),


        (
            em_optimization, 
            25, 100, 100,
            0., 0.,
            {},
            '20news_25t_base_0_0.pkl'
        ),
        (
            artm_thetaless_em_optimization, 
            25, 100, 100,
            0., 0.,
            {'use_B_cheat': False},
            '20news_25t_artm_0_0.pkl'
        ),
        (
            artm_thetaless_em_optimization, 
            25, 100, 100,
            0., 0.,
            {'use_B_cheat': True},
            '20news_25t_artm_0_0_cheat.pkl'
        ),
        (
            em_optimization, 
            25, 100, 100,
            -0.1, 0.,
            {},
            '20news_25t_base_-0.1_0.pkl'
        ),
        (
            artm_thetaless_em_optimization, 
            25, 100, 100,
            -0.1, 0.,
            {'use_B_cheat': False},
            '20news_25t_artm_-0.1_0.pkl'
        ),
        (
            artm_thetaless_em_optimization, 
            25, 100, 100,
            -0.1, 0.,
            {'use_B_cheat': True},
            '20news_25t_artm_-0.1_0_cheat.pkl'
        ),
        (
            em_optimization, 
            25, 100, 100,
            0.1, 0.,
            {},
            '20news_25t_base_+0.1_0.pkl'
        ),
        (
            artm_thetaless_em_optimization, 
            25, 100, 100,
            0.1, 0.,
            {'use_B_cheat': False},
            '20news_25t_artm_+0.1_0.pkl'
        ),
        (
            artm_thetaless_em_optimization, 
            25, 100, 100,
            0.1, 0.,
            {'use_B_cheat': True},
            '20news_25t_artm_+0.1_0_cheat.pkl'
        ),
        (
            em_optimization, 
            25, 100, 100,
            0., -0.1,
            {},
            '20news_25t_base_0_-0.1.pkl'
        ),
        (
            artm_thetaless_em_optimization, 
            25, 100, 100,
            0., -0.1,
            {'use_B_cheat': False},
            '20news_25t_artm_0_-0.1.pkl'
        ),
        (
            artm_thetaless_em_optimization, 
            25, 100, 100,
            0., -0.1,
            {'use_B_cheat': True},
            '20news_25t_artm_0_-0.1_cheat.pkl'
        )
    ]

    Pool(processes=10).map(perform_experiment, args_list)
