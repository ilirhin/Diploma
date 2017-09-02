# coding: utf-8
#coding: utf-8
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from numpy_artm import *
import pickle


def perform_extended_lda(
    T, phi_alpha, theta_alpha, seed, 
    n_dw_matrix, phi_zero_init=None, 
    theta_zero_init=None
):
    D, W = n_dw_matrix.shape
    np.random.seed(seed)

    phi_matrix = np.random.uniform(size=(T, W)).astype(np.float64)
    if phi_zero_init is not None:
        phi_matrix[phi_zero_init < 1e-20] = 0.
    phi_matrix /= np.sum(phi_matrix, axis=1)[:, np.newaxis]

    theta_matrix = np.random.uniform(size=(D, T)).astype(np.float64)
    if theta_zero_init is not None:
        theta_zero_init[theta_zero_init < 1e-20] = 0.
    theta_matrix /= np.sum(theta_matrix, axis=1)[:, np.newaxis]

    regularization_list = np.zeros(200, dtype=object)
    regularization_list[:] = create_reg_lda(phi_alpha, theta_alpha)

    phi, theta = em_optimization(
        n_dw_matrix=n_dw_matrix, 
        phi_matrix=phi_matrix,
        theta_matrix=theta_matrix,
        regularization_list=regularization_list,
        iters_count=100,
        iteration_callback=None,
        params={}
    )
    
    return phi, theta


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
n_dw_matrix, _, _, _ = prepare_dataset(dataset)
_train_perplexity = artm_calc_perplexity_factory(n_dw_matrix) 


print 'Original PLSA'

phis = []
perplexities = []
for seed in xrange(3):
    print seed
    phi, theta = perform_extended_lda(10, 0., 0., seed, n_dw_matrix)
    phis.append(phi.flatten())
    perplexities.append(_train_perplexity(phi, theta))


with open('check_uniqueness/plsa.pkl', 'w') as f:
    pickle.dump({
        'init_phi': None,
        'init_theta': None,
        'perplexities': perplexities,
        'phis': phis
    }, f)


print 'Full initialized PLSA'

init_phi, init_theta = perform_extended_lda(10, -0.1, 0., 42, n_dw_matrix)
new_init_phi, new_init_theta = perform_extended_lda(10, 0., 0., 42, n_dw_matrix, phi_zero_init=init_phi, theta_zero_init=init_theta)

phis = []
perplexities = []
for seed in xrange(3):
    print seed
    phi, theta = perform_extended_lda(10, 0., 0., seed, n_dw_matrix, phi_zero_init=new_init_phi, theta_zero_init=new_init_theta)
    phis.append(phi.flatten())
    perplexities.append(_train_perplexity(phi, theta))


with open('check_uniqueness/full_initialized_plsa.pkl', 'w') as f:
    pickle.dump({
        'init_phi': new_init_phi,
        'init_theta': new_init_theta,
        'perplexities': perplexities,
        'phis': phis
    }, f)


print 'Syntetic PLSA'

m = np.dot(new_init_theta, new_init_phi)
origin_phi = np.array(new_init_phi)
origin_theta = np.array(new_init_theta)
print np.sum(np.isnan(m))
m[np.isnan(m)] = 0.
new_n_dw_matrix = scipy.sparse.csr_matrix(m)
_train_perplexity = artm_calc_perplexity_factory(new_n_dw_matrix) 


phis = []
perplexities = []
for seed in xrange(100):
    print seed
    phi, theta = perform_extended_lda(10, 0., 0., seed, new_n_dw_matrix)
    phis.append(phi.flatten())
    perplexities.append(_train_perplexity(phi, theta))

with open('check_uniqueness/syntetic_plsa.pkl', 'w') as f:
    pickle.dump({
        'init_phi': origin_phi,
        'init_theta': origin_theta,
        'perplexities': perplexities,
        'phis': phis
    }, f)


print 'Full initialized syntetic PLSA'

phis = []
perplexities = []
for seed in xrange(100):
    print seed
    phi, theta = perform_extended_lda(10, 0., 0., seed, new_n_dw_matrix, phi_zero_init=origin_phi, theta_zero_init=origin_theta)
    phis.append(phi.flatten())
    perplexities.append(external_calculate_perplexity(new_freq_matrix, phi, theta))

with open('check_uniqueness/full_initialized_syntetic_plsa.pkl', 'w') as f:
    pickle.dump({
        'init_phi': origin_phi,
        'init_theta': origin_theta,
        'perplexities': perplexities,
        'phis': phis
    }, f)
