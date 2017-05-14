# coding: utf-8
import numpy as np
import scipy
import scipy.sparse
from sklearn.datasets import fetch_20newsgroups
import gensim
from collections import Counter
import heapq
import nltk
from nltk.corpus import stopwords
from sklearn.cross_validation import cross_val_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import pickle
from sklearn.manifold import TSNE

nltk.download('stopwords')
english_stopwords = set(stopwords.words('english'))

def trivial_p_dwt_processor(p_dwt):
    pass

def create_frac_of_max_p_dwt_processor(frac_size):
    def fun(p_dwt):
        maximums = np.max(p_dwt, axis=1)
        p_dwt[p_dwt < frac_size * maximums[:, np.newaxis]] = 0.
        p_dwt /= (np.sum(p_dwt, axis=1)[:, np.newaxis] + 1e-10)
    return fun


def perform_e_step_update(it_num, freq_matrix, docptr, phi_matrix, theta_matrix, params):
    block_size = params.get('block_size', 1)
    p_dwt_processor = params.get('p_dwt_processor', lambda x: None)
    event_handler = params.get('event_handler', EmptyHandler())
    
    D, W = freq_matrix.shape
    T = phi_matrix.shape[0]
    n_wt, n_dt = np.zeros((W, T)), np.zeros((D, T))
    transposed_phi_matrix = np.transpose(phi_matrix)
    
    indices = freq_matrix.indices
    indptr = freq_matrix.indptr
    data = freq_matrix.data
    
    for block_num in xrange((D + block_size - 1) / block_size):
        block_start = block_num * block_size
        block_finish = min(D, block_start + block_size)
        ind_start, ind_finish = indptr[block_start], indptr[block_finish]
        
        datas = data[ind_start:ind_finish]
        words = indices[ind_start:ind_finish]
        docs = docptr[ind_start:ind_finish]
        
        p_dwt = transposed_phi_matrix[words] * theta_matrix[docs, :]
        p_dwt /= (np.sum(p_dwt, axis=1)[:, np.newaxis] + 1e-20)
        p_dwt_processor(p_dwt)
        p_dwt *= datas[:, np.newaxis]
        
        for doc_num in xrange(block_start, block_finish):
            doc_start, doc_finish = indptr[doc_num], indptr[doc_num + 1]
            doc_p_dwt = p_dwt[(doc_start - ind_start):(doc_finish - ind_start), :]
            n_dt[doc_num, :] += np.sum(doc_p_dwt, axis=0)
            n_wt[indices[doc_start:doc_finish], :] += doc_p_dwt
            
    event = {
        'n_wt': n_wt,
        'n_dt': n_dt,
        'freq_matrix': freq_matrix,
        'phi_matrix': phi_matrix,
        'theta_matrix': theta_matrix,
        'docptr': docptr,
        'it_num': it_num
    }
    event_handler.handle(event)

    return n_wt, n_dt


def launch_em(
    freq_matrix, 
    phi_matrix,
    theta_matrix,
    regularizations_list,
    params_list,
    iters_count=100
):
    phi_matrix = np.copy(phi_matrix)
    theta_matrix = np.copy(theta_matrix)
    docptr = []
    indptr = freq_matrix.indptr
    for doc_num in xrange(D):
        docptr.extend([doc_num] * (indptr[doc_num + 1] - indptr[doc_num]))
    docptr = np.array(docptr)
    
    for it in xrange(iters_count):
        n_wt, n_dt = params_list[it]['method'](it, freq_matrix, docptr, phi_matrix, theta_matrix, params_list[it])
        r_wt, r_dt = regularizations_list[it](n_wt, n_dt, phi_matrix, theta_matrix)
        n_wt = np.maximum(n_wt + r_wt, 0)
        n_dt = np.maximum(n_dt + r_dt, 0)
        n_wt /= np.sum(n_wt, axis=0)
        n_dt /= np.sum(n_dt, axis=1)[:, np.newaxis]
        phi_matrix = np.transpose(n_wt)
        theta_matrix = n_dt
        
    return phi_matrix, theta_matrix


# In[7]:

def trivial_regularization(n_wt, n_dt, phi_matrix, theta_matrix):
    return 0., 0.

def calculate_decorr(phi_matrix):
    aggr_phi = np.sum(phi_matrix, axis=1)
    return np.sum(phi_matrix * (aggr_phi[:, np.newaxis] - phi_matrix))

def create_reg_decorr_naive(tau, theta_alpha=0.):
    def fun (n_wt, n_dt, phi_matrix, theta_matrix):
        aggr_phi = np.sum(phi_matrix, axis=1)
        return - tau * np.transpose(phi_matrix * (aggr_phi[:, np.newaxis] - phi_matrix)), theta_alpha
    return fun

def create_reg_lda(phi_alpha, theta_alpha):
    def fun (n_wt, n_dt, phi_matrix, theta_matrix):
        return phi_alpha, theta_alpha
    return fun

def create_reg_decorr_unbiased(tau, theta_alpha=0.):
    def fun (n_wt, n_dt, phi_matrix, theta_matrix):
        tmp_phi =  n_wt / np.sum(n_wt, axis=0)
        aggr_phi = np.sum(tmp_phi, axis=0)
        return - tau * tmp_phi * (aggr_phi[np.newaxis, :] - tmp_phi), theta_alpha
    return fun

def calculate_likelihood(freq_matrix, docptr, phi_matrix, theta_matrix, block_size=1):
    D, W = freq_matrix.shape
    T = phi_matrix.shape[0]
    transposed_phi_matrix = np.transpose(phi_matrix)
    
    indices = freq_matrix.indices
    indptr = freq_matrix.indptr
    data = freq_matrix.data
    
    res = 0.
    for block_num in xrange((D + block_size - 1) / block_size):
        block_start = block_num * block_size
        block_finish = min(D, block_start + block_size)
        ind_start, ind_finish = indptr[block_start], indptr[block_finish]
        
        datas = data[ind_start:ind_finish]
        words = indices[ind_start:ind_finish]
        docs = docptr[ind_start:ind_finish]
        
        p_dwt = transposed_phi_matrix[words] * theta_matrix[docs, :]
        res += np.sum(np.log(np.sum(p_dwt, axis=1) + 1e-20) * datas)
    
    return res


def external_calculate_likelihood(freq_matrix, phi_matrix, theta_matrix):
    docptr = []
    indptr = freq_matrix.indptr
    for doc_num in xrange(D):
        docptr.extend([doc_num] * (indptr[doc_num + 1] - indptr[doc_num]))
    docptr = np.array(docptr)
    
    return calculate_likelihood(freq_matrix, docptr, phi_matrix, theta_matrix, block_size=50)

def external_calculate_perplexity(freq_matrix, phi_matrix, theta_matrix):
    likelihood = external_calculate_likelihood(freq_matrix, phi_matrix, theta_matrix)
    return np.exp(- likelihood / freq_matrix.sum())

class EmptyLogger(object):
    def iteration(*args):
        pass
    def final_info(self, ):
        pass
    
class DecorrWatcher(object):
    def __init__(self, tau):
        self.tau = tau

    def iteration(self, iter_num, freq_matrix, docptr, phi_matrix, theta_matrix, res):
        print iter_num
        likelihood = calculate_likelihood(freq_matrix, docptr, phi_matrix, theta_matrix, 50)
        decorr = calculate_decorr(phi_matrix)
        print 'L', likelihood
        print 'decorr', decorr
        print 'L + tau R', likelihood - self.tau * decorr
        non_zeros = np.sum(phi_matrix > 1e-20)
        size = phi_matrix.shape[0] * phi_matrix.shape[1]
        print 'Phi non zeros elements', non_zeros, '   fraction', round(1. * non_zeros / size, 2)
        non_zeros = np.sum(theta_matrix > 1e-20)
        size = theta_matrix.shape[0] * theta_matrix.shape[1]
        print 'Theta non zeros elements', non_zeros, '   fraction', round(1. * non_zeros / size, 2)
    def final_info(self):
        pass


class EmptyHandler(object):
    def requires(self, name):
        return False
        
    def handle(self, event):
        pass
    
    def final(self):
        pass
            


# In[8]:

def prepare_dataset(dataset):
    # remove stopwords
    occurences = Counter()
    for i, doc in enumerate(dataset.data):
        tokens = gensim.utils.lemmatize(doc)
        for token in set(tokens):
            occurences[token] += 1
        if i % 500 == 0:
            print 'Processed: ', i, 'documents from', len(dataset.data)
    
    row, col, data = [], [], []
    token_2_num = {}
    for i, doc in enumerate(dataset.data):
        tokens = gensim.utils.lemmatize(doc)
        cnt = Counter()
        for token in tokens:
            word = token.split('/')[0]
            if word not in english_stopwords and 10 <= occurences[token] < len(dataset.data) / 2:
                if token not in token_2_num:
                    token_2_num[token] = len(token_2_num)
                cnt[token_2_num[token]] += 1
        for w, c in cnt.iteritems():
            row.append(i)
            col.append(w)
            data.append(c)
        
    num_2_token = {
        v: k
        for k, v in token_2_num.iteritems()
    }
    print 'Nonzero values:', len(data)
    return scipy.sparse.csr_matrix((data, (row, col))), token_2_num, num_2_token


# In[9]:

def prepare_dataset_with_cooccurences(dataset):
    # remove stopwords
    occurences = Counter()
    cooccurences = Counter()
    for i, doc in enumerate(dataset.data):
        tokens = gensim.utils.lemmatize(doc)
        for token in set(tokens):
            occurences[token] += 1
        if i % 500 == 0:
            print 'Preprocessed: ', i, 'documents from', len(dataset.data)
    
    row, col, data = [], [], []
    token_2_num = {}
    for i, doc in enumerate(dataset.data):
        tokens = gensim.utils.lemmatize(doc)
        cnt = Counter()
        words_set = set()
        for token in tokens:
            word = token.split('/')[0]
            if word not in english_stopwords and 10 <= occurences[token] < len(dataset.data) / 2:
                if token not in token_2_num:
                    token_2_num[token] = len(token_2_num)
                words_set.add(token_2_num[token])
                cnt[token_2_num[token]] += 1
        for w, c in cnt.iteritems():
            row.append(i)
            col.append(w)
            data.append(c)
            
        for w1 in words_set:
            for w2 in words_set:
                cooccurences[(w1, w2)] += 1
                
        if i % 500 == 0:
            print 'Processed: ', i, 'documents from', len(dataset.data)
        
    num_2_token = {
        v: k
        for k, v in token_2_num.iteritems()
    }
    print 'Nonzero values:', len(data)
    return scipy.sparse.csr_matrix((data, (row, col))), token_2_num, num_2_token, cooccurences


dataset = fetch_20newsgroups(
    subset='all',
    categories=['sci.electronics', 'sci.med', 'sci.space'],
    remove=('headers', 'footers', 'quotes')
)

origin_freq_matrix, token_2_num, num_2_token = prepare_dataset(dataset)


def investigate_phi_uniqueness(phi):
    T, W = phi.shape
    for t in xrange(T):
        matrix = phi
        positions = matrix[t, :] == 0.
        topics = [x for x in xrange(T) if x != t]

        print 'Topic', t
        print '\t', np.sum(positions), 'zeros'
        if np.sum(positions) != 0:
            print '\tSubmatrix rank', np.linalg.matrix_rank(matrix[np.ix_(topics, positions)])
            eigen_values = np.linalg.svd(matrix[np.ix_(topics, positions)])[1]
            print '\tEigen values:', eigen_values
            max_val = np.min(np.linalg.svd(matrix[topics, :])[1])
            print '\tUniqueness measure:', min(eigen_values)
            print '\tNormalized uniqueness measure:', min(eigen_values) / max_val
        else:
            print '\tUniqueness measure:', 0.


def calc_phi_uniqueness_measures(phi):
    T, W = phi.shape
    res = []
    nres = []
    for t in xrange(T):
        positions = phi[t, :] == 0.
        topics = [x for x in xrange(T) if x != t]
        if np.sum(positions) == 0:
            res.append(0.)
            nres.append(0.)
        else:
            rank = np.linalg.matrix_rank(phi[np.ix_(topics, positions)])
            if rank == T - 1:
                max_val = np.min(np.linalg.svd(phi[topics, :])[1])
                curr_val = np.min(np.linalg.svd(phi[np.ix_(topics, positions)])[1])
                res.append(curr_val)
                nres.append(curr_val / max_val)
            else:
                res.append(0.)
                nres.append(0.)
    return res, nres


D, W = origin_freq_matrix.shape

def perform_extended_lda(
    T, words_alpha, docs_alpha, seed=42, 
    freq_matrix=origin_freq_matrix, phi_zero_init=None, 
    theta_zero_init=None
):
    D, W = freq_matrix.shape

    np.random.seed(seed)

    phi_matrix = np.random.uniform(size=(T, W)).astype(np.float64)
    if phi_zero_init is not None:
        phi_matrix[phi_zero_init < 1e-20] = 0.
    phi_matrix /= np.sum(phi_matrix, axis=1)[:, np.newaxis]

    theta_matrix = np.random.uniform(size=(D, T)).astype(np.float64)
    if theta_zero_init is not None:
        theta_zero_init[theta_zero_init < 1e-20] = 0.
    theta_matrix /= np.sum(theta_matrix, axis=1)[:, np.newaxis]

    no_selection_params = {
        'method': perform_e_step_update,
        'block_size': 50,
        'p_dwt_processor': trivial_p_dwt_processor
    }

    regularizations_list = np.zeros(50, dtype=object)
    params_list = np.zeros(50, dtype=object)

    regularizations_list[:10] = trivial_regularization
    regularizations_list[10:] = create_reg_lda(words_alpha, docs_alpha)
    params_list[:] = no_selection_params

    phi, theta = launch_em(
        freq_matrix=freq_matrix, 
        phi_matrix=phi_matrix,
        theta_matrix=theta_matrix,
        regularizations_list=regularizations_list,
        params_list=params_list,
        iters_count=50
    )
    
    return phi, theta


phis = []
perplexities = []
for seed in xrange(300):
    print seed
    phi, theta = perform_extended_lda(5, 0., 0., seed=seed)
    phis.append(phi.flatten())
    perplexities.append(external_calculate_perplexity(origin_freq_matrix, phi, theta))


with open('check_uniqueness/plsa.pkl', 'w') as f:
    pickle.dump({
        'init_phi': None,
        'init_theta': None,
        'perplexities': perplexities,
        'phis': phis
    }, f)

# # Full initialized PLSA

init_phi, init_theta = perform_extended_lda(5, -0.1, 0., seed=42)
new_init_phi, new_init_theta = perform_extended_lda(5, 0., 0., seed=42, phi_zero_init=init_phi, theta_zero_init=init_theta)

phis = []
perplexities = []
for seed in xrange(300):
    print seed
    phi, theta = perform_extended_lda(5, 0., 0., seed=seed, phi_zero_init=new_init_phi, theta_zero_init=new_init_theta)
    phis.append(phi.flatten())
    perplexities.append(external_calculate_perplexity(origin_freq_matrix, phi, theta))


with open('check_uniqueness/full_initialized_plsa.pkl', 'w') as f:
    pickle.dump({
        'init_phi': new_init_phi,
        'init_theta': new_init_theta,
        'perplexities': perplexities,
        'phis': phis
    }, f)


# # Syntetic PLSA

m = np.dot(new_init_theta, new_init_phi)
origin_phi = np.array(new_init_phi)
origin_theta = np.array(new_init_theta)
print np.sum(np.isnan(m))
m[np.isnan(m)] = 0.
new_freq_matrix = scipy.sparse.csr_matrix(m)


phis = []
perplexities = []
for seed in xrange(60):
    print seed
    phi, theta = perform_extended_lda(5, 0., 0., seed=seed, freq_matrix=new_freq_matrix)
    phis.append(phi.flatten())
    perplexities.append(external_calculate_perplexity(new_freq_matrix, phi, theta))

with open('check_uniqueness/syntetic_plsa.pkl', 'w') as f:
    pickle.dump({
        'init_phi': origin_phi,
        'init_theta': origin_theta,
        'perplexities': perplexities,
        'phis': phis
    }, f)


# # Full initialized syntetic PLSA

phis = []
perplexities = []
for seed in xrange(60):
    print seed
    phi, theta = perform_extended_lda(5, 0., 0., seed=seed, phi_zero_init=origin_phi, theta_zero_init=origin_theta, freq_matrix=new_freq_matrix)
    phis.append(phi.flatten())
    perplexities.append(external_calculate_perplexity(new_freq_matrix, phi, theta))

with open('check_uniqueness/full_initialized_syntetic_plsa.pkl', 'w') as f:
    pickle.dump({
        'init_phi': origin_phi,
        'init_theta': origin_theta,
        'perplexities': perplexities,
        'phis': phis
    }, f)
