#coding=utf-8
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
from sklearn.manifold import TSNE
import pickle

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


import random

def prepare_train_test_dataset(dataset):
    random.seed(44)
    # remove stopwords
    occurences = Counter()
    for i, doc in enumerate(dataset.data):
        tokens = gensim.utils.lemmatize(doc)
        for token in set(tokens):
            occurences[token] += 1
        if i % 500 == 0:
            print 'Processed: ', i, 'documents from', len(dataset.data)
    
    train_row, train_col, train_data = [], [], []
    test_row, test_col, test_data = [], [], []
    token_2_num = {}
    for i, doc in enumerate(dataset.data):
        tokens = gensim.utils.lemmatize(doc)
        cnt_train = Counter()
        cnt_test = Counter()
        for token in tokens:
            word = token.split('/')[0]
            if word not in english_stopwords and 10 <= occurences[token] < len(dataset.data) / 2:
                if token not in token_2_num:
                    token_2_num[token] = len(token_2_num)
                if random.random() < 0.075:
                    cnt_test[token_2_num[token]] += 1
                else:
                    cnt_train[token_2_num[token]] += 1
        for w, c in cnt_test.iteritems():
            test_row.append(i)
            test_col.append(w)
            test_data.append(c)
        for w, c in cnt_train.iteritems():
            train_row.append(i)
            train_col.append(w)
            train_data.append(c)
        
    num_2_token = {
        v: k
        for k, v in token_2_num.iteritems()
    }
    return (
        scipy.sparse.csr_matrix((train_data, (train_row, train_col))),
        scipy.sparse.csr_matrix((test_data, (test_row, test_col))),
        token_2_num,
        num_2_token
    )


dataset = fetch_20newsgroups(
    subset='all',
    categories=['sci.electronics', 'sci.med', 'sci.space', 'sci.crypt', 'rec.autos', 'rec.sport.baseball', 'rec.sport.hockey'],
    remove=('headers', 'footers', 'quotes')
)

train_origin_freq_matrix, test_origin_freq_matrix, _, _ = prepare_train_test_dataset(dataset)
D, W = train_origin_freq_matrix.shape


print train_origin_freq_matrix.shape
print test_origin_freq_matrix.shape


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

def perform_lda(T, words_alpha, docs_alpha, seed=42):
    D, W = train_origin_freq_matrix.shape

    np.random.seed(seed)

    phi_matrix = np.random.uniform(size=(T, W)).astype(np.float64)
    phi_matrix /= np.sum(phi_matrix, axis=1)[:, np.newaxis]

    theta_matrix = np.random.uniform(size=(D, T)).astype(np.float64)
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
        freq_matrix=train_origin_freq_matrix, 
        phi_matrix=phi_matrix,
        theta_matrix=theta_matrix,
        regularizations_list=regularizations_list,
        params_list=params_list,
        iters_count=50
    )
    
    return phi, theta


font = {'family' : 'normal',
        'size'   : 12}

plt.rc('font', **font)


topics_values = []

train_perplexity_values = []
test_perplexity_values = []
min_ums_values = []
min_nums_values = []
max_ums_values = []
max_nums_values = []
avg_ums_values = []
avg_nums_values = []

train_perplexity_values_err = []
test_perplexity_values_err = []
min_ums_values_err = []
min_nums_values_err = []
max_ums_values_err = []
max_nums_values_err = []
avg_ums_values_err = []
avg_nums_values_err = []

for T in xrange(5, 36, 2):
    print T
    min_ums_value = [] 
    min_nums_value = [] 
    max_ums_value = [] 
    max_nums_value = [] 
    avg_ums_value = [] 
    avg_nums_value = [] 
    train_perplexity_value = []
    test_perplexity_value = []
    
    for seed in xrange(30):
        phi, theta = perform_lda(T, - 0.1, 0., seed=seed)
        theta[np.isnan(theta)] = 1e-20
        theta /= np.sum(theta, axis=1)[:, np.newaxis]
        
        dums, dnums = calc_phi_uniqueness_measures(phi)
        
        min_ums_value.append(np.min(dums))
        min_nums_value.append(np.min(dnums)) 
        max_ums_value.append(np.max(dums)) 
        max_nums_value.append(np.max(dnums))
        avg_ums_value.append(np.mean(dums))
        avg_nums_value.append(np.mean(dnums))
        train_perplexity_value.append(external_calculate_perplexity(train_origin_freq_matrix, phi, theta))
        test_perplexity_value.append(external_calculate_perplexity(test_origin_freq_matrix, phi, theta))
    
    topics_values.append(T)
    n = len(min_ums_value)

    train_perplexity_values.append(np.mean(train_perplexity_value))
    test_perplexity_values.append(np.mean(test_perplexity_value))
    min_ums_values.append(np.mean(min_ums_value))
    min_nums_values.append(np.mean(min_nums_value))
    max_ums_values.append(np.mean(max_ums_value))
    max_nums_values.append(np.mean(max_nums_value))
    avg_ums_values.append(np.mean(avg_ums_value))
    avg_nums_values.append(np.mean(avg_nums_value))

    train_perplexity_values_err.append(3. / np.sqrt(n) * np.std(train_perplexity_value))
    test_perplexity_values_err.append(3. / np.sqrt(n) * np.std(test_perplexity_value))
    min_ums_values_err.append(3. / np.sqrt(n) * np.std(min_ums_value))
    min_nums_values_err.append(3. / np.sqrt(n) * np.std(min_nums_value))
    max_ums_values_err.append(3. / np.sqrt(n) * np.std(max_ums_value))
    max_nums_values_err.append(3. / np.sqrt(n) * np.std(max_nums_value))
    avg_ums_values_err.append(3. / np.sqrt(n) * np.std(avg_ums_value))
    avg_nums_values_err.append(3. / np.sqrt(n) * np.std(avg_nums_value))

    print min_nums_values[-5:]
    print min_nums_values_err[-5:]
    print train_perplexity_values[-5:]
    print train_perplexity_values_err[-5:]
    print test_perplexity_values[-5:]
    print test_perplexity_values_err[-5:]
    print ''
    print ''

with open('/home/tylorn/uniqueness_perp/topics_dependency_origin_big.pkl', 'w') as f:
    pickle.dump({
        'topics_values': topics_values,
        
        'avg_nums_values': avg_nums_values,
        'min_nums_values': min_nums_values,
        'max_nums_values': max_nums_values,
        'avg_ums_values': avg_ums_values,
        'min_ums_values': min_ums_values,
        'max_ums_values': max_ums_values,
        'train_perplexity_values': train_perplexity_values,
        'test_perplexity_values': test_perplexity_values,

        'avg_nums_values_err': avg_nums_values_err,
        'min_nums_values_err': min_nums_values_err,
        'max_nums_values_err': max_nums_values_err,
        'avg_ums_values_err': avg_ums_values_err,
        'min_ums_values_err': min_ums_values_err,
        'max_ums_values_err': max_ums_values_err,
        'train_perplexity_values_err': train_perplexity_values_err,
        'test_perplexity_values_err': test_perplexity_values_err
    }, f)
