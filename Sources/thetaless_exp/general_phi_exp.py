import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import argparse
import pickle
from numpy_artm import *


def perform_phi_experiment(
    optimization_method,
    T, iters_count,
    phi_alpha, theta_alpha, 
    train_n_dw_matrix, test_n_dw_matrix,
    token_2_num, num_2_token, doc_targets,
    doc_occurences, doc_cooccurences,
    samples=10
):
    D, W = train_n_dw_matrix.shape

    train_perplexities = []
    test_perplexities = []
    sparsities = []
    theta_sparsities = []
    topic_correlations = []
    avg_top5_pmis = []
    avg_top10_pmis = []
    avg_top20_pmis = []
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
        kernel_avg_size = []
        kernel_avg_jacard = []

        np.random.seed(seed)

        phi_matrix = np.random.uniform(size=(T, W)).astype(np.float64)
        phi_matrix /= np.sum(phi_matrix, axis=1)[:, np.newaxis]

        theta_matrix = np.ones(shape=(D, T)).astype(np.float64)
        theta_matrix /= np.sum(theta_matrix, axis=1)[:, np.newaxis]

        regularization_list = np.zeros(200, dtype=object)
        regularization_list[:] = create_reg_lda(0., 0.)

        _train_perplexity = artm_calc_perplexity_factory(train_n_dw_matrix) 
        _test_perplexity = artm_calc_perplexity_factory(test_n_dw_matrix)
        _top5_pmi = artm_calc_pmi_top_factory(doc_occurences, doc_cooccurences, train_n_dw_matrix.shape[0], 5)
        _top10_pmi = artm_calc_pmi_top_factory(doc_occurences, doc_cooccurences, train_n_dw_matrix.shape[0], 10)
        _top20_pmi = artm_calc_pmi_top_factory(doc_occurences, doc_cooccurences, train_n_dw_matrix.shape[0], 20)

        def callback(it, phi, theta):
            train_perplexity.append(_train_perplexity(phi, theta))
            test_perplexity.append(_test_perplexity(phi, theta))
            topic_correlation.append(artm_calc_topic_correlation(phi))
            sparsity.append(1. * np.sum(phi < 1e-20) / np.sum(phi >= 0))
            theta_sparsity.append(1. * np.sum(theta < 0.01) / np.sum(theta >= 0))
            avg_top5_pmi.append(_top5_pmi(phi))
            avg_top10_pmi.append(_top10_pmi(phi))
            avg_top20_pmi.append(_top20_pmi(phi))
            kernel_avg_size.append(np.mean(artm_get_kernels_sizes(phi)))
            kernel_avg_jacard.append(artm_get_avg_pairwise_kernels_jacards(phi))

        phi, theta = optimization_method(
            n_dw_matrix=train_n_dw_matrix, 
            phi_matrix=phi_matrix,
            theta_matrix=theta_matrix,
            regularization_list=regularization_list,
            iters_count=iters_count,
            iteration_callback=callback
        )
        
        train_perplexities.append(train_perplexity)
        test_perplexities.append(test_perplexity)
        sparsities.append(sparsity)
        theta_sparsities.append(theta_sparsity)
        topic_correlations.append(topic_correlation)
        avg_top5_pmis.append(avg_top5_pmi)
        avg_top10_pmis.append(avg_top10_pmi)
        avg_top20_pmis.append(avg_top20_pmi)

    return {
        'train_perplexities': train_perplexities,
        'test_perplexities': test_perplexities,
        'sparsities': sparsities,
        'theta_sparsities': theta_sparsities,
        'topic_correlations': topic_correlations,
        'avg_top5_pmis': avg_top5_pmis,
        'avg_top10_pmis': avg_top10_pmis,
        'avg_top20_pmis': avg_top20_pmis
    }
  

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Thetaless experiment')
    parser.add_argument('--topics', type=int, required=True, help='number of topics')
    parser.add_argument('--iters', type=int, required=True, help='iters_count')
    parser.add_argument('--output-path', type=str, required=True, help='output_path')

    parser.add_argument('--samples', type=int, required=False, help='number of samples')
    parser.set_defaults(samples=10)

    parser.add_argument('--phi-alpha', type=float, required=False, help='phi alpha')
    parser.set_defaults(phi_alpha=0.)
    
    parser.add_argument('--theta-alpha', type=float, required=False, help='theta alpha')
    parser.set_defaults(theta_alpha=0.)

    parser.add_argument('--train-test-split', type=float, required=False, help='train_test_split')
    parser.set_defaults(train_test_split=0.8)

    args = parser.parse_args()

    dataset = fetch_20newsgroups(
        subset='all',
        categories=['sci.electronics', 'sci.med', 'sci.space', 'sci.crypt', 'rec.sport.baseball', 'rec.sport.hockey'],
        remove=('headers', 'footers', 'quotes')
    )
    train_n_dw_matrix, test_n_dw_matrix, token_2_num, num_2_token, doc_targets, doc_occurences, doc_cooccurences = prepare_dataset(dataset, calc_cooccurences=True, train_test_split=args.train_test_split)

    base_res = perform_phi_experiment(
        optimization_method=em_optimization,
        T=args.topics, 
        iters_count=args.iters,
        phi_alpha=args.phi_alpha, 
        theta_alpha=args.theta_alpha, 
        train_n_dw_matrix=train_n_dw_matrix, 
        test_n_dw_matrix=test_n_dw_matrix,
        token_2_num=token_2_num, 
        num_2_token=num_2_token, 
        doc_targets=doc_targets,
        doc_occurences=doc_occurences, 
        doc_cooccurences=doc_cooccurences,
        samples=args.samples
    )

    artm_res = perform_phi_experiment(
        optimization_method=artm_thetaless_em_optimization,
        T=args.topics, 
        iters_count=args.iters,
        phi_alpha=args.phi_alpha, 
        theta_alpha=args.theta_alpha, 
        train_n_dw_matrix=train_n_dw_matrix, 
        test_n_dw_matrix=test_n_dw_matrix,
        token_2_num=token_2_num, 
        num_2_token=num_2_token, 
        doc_targets=doc_targets,
        doc_occurences=doc_occurences, 
        doc_cooccurences=doc_cooccurences,
        samples=args.samples
    )

    with open(args.output_path, 'w') as f:
        pickle.dump({
            'base_res': base_res,
            'artm_res': artm_res
        }, f)   
