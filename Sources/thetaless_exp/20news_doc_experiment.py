#coding: utf-8
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from numpy_artm import *
from multiprocessing import Pool, Manager
import pickle
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


def perform_experiment((
    optimization_method,
    T, iters_count, samples,
    phi_alpha, theta_alpha, 
    params,
    output_path
)):
    dataset_train = fetch_20newsgroups(
        subset='train',
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
    n_dw_matrix_doc_train, token_2_num_doc_train, num_2_token_doc_train, doc_targets_doc_train = prepare_dataset(dataset_train)

    dataset_test = fetch_20newsgroups(
        subset='test',
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
    n_dw_matrix_doc_test, token_2_num_doc_test, num_2_token_doc_test, doc_targets_doc_test = prepare_dataset(dataset_test, token_2_num=token_2_num_doc_train)

    res_plsa_not_const_phi = []
    res_plsa_const_phi = []
    res_artm_thetaless = []
    cv_fold_scores = []
    cv_test_scores = []

    for seed in xrange(samples):
        print seed

        D, W = n_dw_matrix_doc_train.shape

        plsa_not_const_phi = []
        plsa_const_phi = []
        artm_thetaless = []

        np.random.seed(seed)

        phi_matrix = np.random.uniform(size=(T, W)).astype(np.float64)
        phi_matrix /= np.sum(phi_matrix, axis=1)[:, np.newaxis]

        theta_matrix = np.ones(shape=(D, T)).astype(np.float64)
        theta_matrix /= np.sum(theta_matrix, axis=1)[:, np.newaxis]

        regularization_list = np.zeros(200, dtype=object)
        regularization_list[:] = create_reg_lda(phi_alpha, theta_alpha)

        phi, theta = optimization_method(
            n_dw_matrix=n_dw_matrix_doc_train, 
            phi_matrix=phi_matrix,
            theta_matrix=theta_matrix,
            regularization_list=regularization_list,
            iters_count=iters_count,
            params=params
        )

        best_C, best_gamma, cv_fold_score, cv_test_score = svm_score(theta, doc_targets_doc_train, verbose=False)
        cv_fold_scores.append(cv_fold_score)
        cv_test_scores.append(cv_test_score)

        algo = SVC(C=best_C, gamma=best_gamma).fit(theta, doc_targets_doc_train)

        D, _ = n_dw_matrix_doc_test.shape
        theta_matrix = np.ones(shape=(D, T)).astype(np.float64)
        theta_matrix /= np.sum(theta_matrix, axis=1)[:, np.newaxis]

        em_optimization(
            n_dw_matrix=n_dw_matrix_doc_test, 
            phi_matrix=phi,
            theta_matrix=theta_matrix,
            regularization_list=regularization_list,
            iters_count=10,
            iteration_callback=lambda it, phi, theta: plsa_not_const_phi.append(accuracy_score(algo.predict(theta), doc_targets_doc_test))
        )

        em_optimization(
            n_dw_matrix=n_dw_matrix_doc_test, 
            phi_matrix=phi,
            theta_matrix=theta_matrix,
            regularization_list=regularization_list,
            iters_count=10,
            const_phi=True,
            iteration_callback=lambda it, phi, theta: plsa_const_phi.append(accuracy_score(algo.predict(theta), doc_targets_doc_test))
        )

        artm_thetaless_em_optimization(
            n_dw_matrix=n_dw_matrix_doc_test, 
            phi_matrix=phi,
            regularization_list=regularization_list,
            iters_count=10,
            iteration_callback=lambda it, phi, theta: artm_thetaless.append(accuracy_score(algo.predict(theta), doc_targets_doc_test))
        )

        res_plsa_not_const_phi.append(plsa_not_const_phi)
        res_plsa_const_phi.append(plsa_const_phi)
        res_artm_thetaless.append(artm_thetaless)


    with open(output_path, 'w') as f:
        pickle.dump({
            'res_plsa_not_const_phi': res_plsa_not_const_phi,
            'res_plsa_const_phi': res_plsa_const_phi,
            'res_artm_thetaless': res_artm_thetaless,
            'cv_fold_scores': cv_fold_scores,
            'cv_test_scores': cv_test_scores
        }, f)

if __name__ == '__main__':
    args_list = [
        (
            em_optimization, 
            10, 100, 1000,
            0., 0.,
            {},
            'docs_20news_10t_base_0_0.pkl'
        ),
        (
            artm_thetaless_em_optimization, 
            10, 100, 1000,
            0., 0.,
            {'use_B_cheat': False},
            'docs_20news_10t_artm_0_0.pkl'
        ),
        (
            artm_thetaless_em_optimization, 
            10, 100, 1000,
            0., 0.,
            {'use_B_cheat': True},
            'docs_20news_10t_artm_0_0_cheat.pkl'
        ),
        (
            em_optimization, 
            10, 100, 1000,
            -0.1, 0.,
            {},
            'docs_20news_10t_base_-0.1_0.pkl'
        ),
        (
            artm_thetaless_em_optimization, 
            10, 100, 1000,
            -0.1, 0.,
            {'use_B_cheat': False},
            'docs_20news_10t_artm_-0.1_0.pkl'
        ),
        (
            artm_thetaless_em_optimization, 
            10, 100, 1000,
            -0.1, 0.,
            {'use_B_cheat': True},
            'docs_20news_10t_artm_-0.1_0_cheat.pkl'
        ),
        (
            em_optimization, 
            10, 100, 1000,
            0.1, 0.,
            {},
            'docs_20news_10t_base_+0.1_0.pkl'
        ),
        (
            artm_thetaless_em_optimization, 
            10, 100, 1000,
            0.1, 0.,
            {'use_B_cheat': False},
            'docs_20news_10t_artm_+0.1_0.pkl'
        ),
        (
            artm_thetaless_em_optimization, 
            10, 100, 1000,
            0.1, 0.,
            {'use_B_cheat': True},
            'docs_20news_10t_artm_+0.1_0_cheat.pkl'
        ),
        (
            em_optimization, 
            10, 100, 1000,
            0., -0.1,
            {},
            'docs_20news_10t_base_0_-0.1.pkl'
        ),
        (
            artm_thetaless_em_optimization, 
            10, 100, 1000,
            0., -0.1,
            {'use_B_cheat': False},
            'docs_20news_10t_artm_0_-0.1.pkl'
        ),
        (
            artm_thetaless_em_optimization, 
            10, 100, 1000,
            0., -0.1,
            {'use_B_cheat': True},
            'docs_20news_10t_artm_0_-0.1_cheat.pkl'
        ),


        (
            em_optimization, 
            25, 100, 1000,
            0., 0.,
            {},
            'docs_20news_25t_base_0_0.pkl'
        ),
        (
            artm_thetaless_em_optimization, 
            25, 100, 1000,
            0., 0.,
            {'use_B_cheat': False},
            'docs_20news_25t_artm_0_0.pkl'
        ),
        (
            artm_thetaless_em_optimization, 
            25, 100, 1000,
            0., 0.,
            {'use_B_cheat': True},
            'docs_20news_25t_artm_0_0_cheat.pkl'
        ),
        (
            em_optimization, 
            25, 100, 1000,
            -0.1, 0.,
            {},
            'docs_20news_25t_base_-0.1_0.pkl'
        ),
        (
            artm_thetaless_em_optimization, 
            25, 100, 1000,
            -0.1, 0.,
            {'use_B_cheat': False},
            'docs_20news_25t_artm_-0.1_0.pkl'
        ),
        (
            artm_thetaless_em_optimization, 
            25, 100, 1000,
            -0.1, 0.,
            {'use_B_cheat': True},
            'docs_20news_25t_artm_-0.1_0_cheat.pkl'
        ),
        (
            em_optimization, 
            25, 100, 1000,
            0.1, 0.,
            {},
            'docs_20news_25t_base_+0.1_0.pkl'
        ),
        (
            artm_thetaless_em_optimization, 
            25, 100, 1000,
            0.1, 0.,
            {'use_B_cheat': False},
            'docs_20news_25t_artm_+0.1_0.pkl'
        ),
        (
            artm_thetaless_em_optimization, 
            25, 100, 1000,
            0.1, 0.,
            {'use_B_cheat': True},
            'docs_20news_25t_artm_+0.1_0_cheat.pkl'
        ),
        (
            em_optimization, 
            25, 100, 1000,
            0., -0.1,
            {},
            'docs_20news_25t_base_0_-0.1.pkl'
        ),
        (
            artm_thetaless_em_optimization, 
            25, 100, 1000,
            0., -0.1,
            {'use_B_cheat': False},
            'docs_20news_25t_artm_0_-0.1.pkl'
        ),
        (
            artm_thetaless_em_optimization, 
            25, 100, 1000,
            0., -0.1,
            {'use_B_cheat': True},
            'docs_20news_25t_artm_0_-0.1_cheat.pkl'
        )
    ]

    Pool(processes=10).map(perform_experiment, args_list)
