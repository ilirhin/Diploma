# coding: utf-8

import numpy as np
from numpy.core.umath_tests import inner1d
import scipy
import scipy.sparse
from sklearn.datasets import fetch_20newsgroups
import gensim
from collections import Counter
from collections import defaultdict
import heapq
import nltk
import random
from nltk.corpus import stopwords
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import time


class LogFunction(object):
    def calc(self, x):
        return np.log(x + 1e-20)
    def calc_der(self, x):
        return 1. / (x + 1e-20)
    

class IdFunction(object):
    def calc(self, x):
        return x + 1e-20
    def calc_der(self, x):
        return np.ones_like(x)
    

class SquareFunction(object):
    def calc(self, x):
        return (x + 1e-20) ** 2
    def calc_der(self, x):
        return 2. * (x + 1e-20) ** 2
    

class CubeLogFunction(object):
    def calc(self, x):
        return np.log(x + 1e-20) ** 3
    def calc_der(self, x):
        return 3. * np.log(x + 1e-20) ** 2 / (x + 1e-20)
    

class SquareLogFunction(object):
    def calc(self, x):
        return np.log(x + 1e-20) * np.abs(np.log(x + 1e-20))
    def calc_der(self, x):
        return 2. * np.abs(np.log(x + 1e-20)) / (x + 1e-20)

    
class FiveLogFunction(object):
    def calc(self, x):
        return np.log(x + 1e-20) ** 5
    def calc_der(self, x):
        return 5. * np.log(x + 1e-20) ** 4 / (x + 1e-20)
    

class CubeRootLogFunction(object):
    def calc(self, x):
        return np.cbrt(np.log(x + 1e-20))
    def calc_der(self, x):
        return 1. / 3 / (np.cbrt(np.log(x + 1e-20)) ** 2) / (x + 1e-20)
    
    
class SquareRootLogFunction(object):
    def calc(self, x):
        return np.sqrt(- np.log(x + 1e-20))
    def calc_der(self, x):
        return 1. / 2. / np.sqrt(- np.log(x + 1e-20)) / (x + 1e-20)
    

class ExpFunction(object):
    def calc(self, x):
        return np.exp(x)
    def calc_der(self, x):
        return np.exp(x)

    
class EntropyFunction(object):
    def calc(self, x):
        return (np.log(x + 1e-20) + 50.) * (x + 1e-20)
    def calc_der(self, x):
        return np.log(x + 1e-20) + 50.


def trivial_regularization(n_tw, n_dt):
    return np.zeros_like(n_tw), np.zeros_like(n_dt)


def create_reg_decorr(tau, theta_alpha=0.):
    def fun(n_tw, n_dt):
        phi_matrix = n_tw / np.sum(n_tw, axis=1)[:, np.newaxis]
        theta_matrix = n_dt / np.sum(n_dt, axis=1)[:, np.newaxis]
        aggr_phi = np.sum(phi_matrix, axis=1)
        return - tau * np.transpose(phi_matrix * (aggr_phi[:, np.newaxis] - phi_matrix)), theta_alpha
    return fun


def create_reg_lda(phi_alpha, theta_alpha):
    def fun (n_tw, n_dt):
        return np.zeros_like(n_tw) + phi_alpha, np.zeros_like(n_dt) + theta_alpha
    return fun


def prepare_dataset(dataset, calc_cooccurences=False, train_test_split=None, token_2_num=None):
    english_stopwords = set(stopwords.words('english'))
    is_token_2_num_provided = token_2_num is not None 
    # remove stopwords
    if not is_token_2_num_provided:
        token_2_num = {}
        occurences = Counter()
        for i, doc in enumerate(dataset.data):
            tokens = gensim.utils.lemmatize(doc)
            for token in set(tokens):
                occurences[token] += 1
            if i % 500 == 0:
                print 'Processed: ', i, 'documents from', len(dataset.data)
    
    row, col, data = [], [], []
    row_test, col_test, data_test = [], [], []
    not_empty_docs_number = 0
    doc_targets = []
    doc_cooccurences = Counter()
    doc_occurences = Counter()
    random_gen = random.Random(42)
    
    for doc, target in zip(dataset.data, dataset.target):
        tokens = gensim.utils.lemmatize(doc)
        cnt = Counter()
        cnt_test = Counter()
        for token in tokens:
            word = token.split('/')[0]
            if not is_token_2_num_provided and word not in english_stopwords and 3 <= occurences[token] and token not in token_2_num:
                token_2_num[token] = len(token_2_num)
            if token in token_2_num:
                if train_test_split is None or random_gen.random() < train_test_split:
                    cnt[token_2_num[token]] += 1
                else:
                    cnt_test[token_2_num[token]] += 1
        
        if len(cnt) > 0 and (train_test_split is None or len(cnt_test) > 0):
            for w, c in cnt.iteritems():
                row.append(not_empty_docs_number)
                col.append(w)
                data.append(c)
                
            for w, c in cnt_test.iteritems():
                row_test.append(not_empty_docs_number)
                col_test.append(w)
                data_test.append(c)
                
            not_empty_docs_number += 1
            doc_targets.append(target)
            
            if calc_cooccurences:
                words = set(cnt.keys() + cnt_test.keys())
                doc_occurences.update(words)
                doc_cooccurences.update({(w1, w2) for w1 in words for w2 in words if w1 != w2})
        
    num_2_token = {
        v: k
        for k, v in token_2_num.iteritems()
    }
    print 'Nonzero values:', len(data)
    if train_test_split is None:
        if calc_cooccurences:
            return scipy.sparse.csr_matrix((data, (row, col))), token_2_num, num_2_token, doc_targets, doc_occurences, doc_cooccurences
        else:
            return scipy.sparse.csr_matrix((data, (row, col))), token_2_num, num_2_token, doc_targets
    else:
        if calc_cooccurences:
            return (
                scipy.sparse.csr_matrix((data, (row, col))),
                scipy.sparse.csr_matrix((data_test, (row_test, col_test))),
                token_2_num,
                num_2_token,
                doc_targets,
                doc_occurences,
                doc_cooccurences
            )
        else:
            return (
                scipy.sparse.csr_matrix((data, (row, col))),
                scipy.sparse.csr_matrix((data_test, (row_test, col_test))),
                token_2_num,
                num_2_token,
                doc_targets
            )
        
        
def prepare_nips_dataset(dataset_path, calc_cooccurences=False, train_test_split=None):
    row, col, data = [], [], []
    row_test, col_test, data_test = [], [], []
    not_empty_docs_number = 0
    doc_targets = []
    doc_cooccurences = Counter()
    doc_occurences = Counter()
    random_gen = random.Random(42)
    token_2_num = {}
    documents = defaultdict(list)

    with open(dataset_path, 'r') as f:
        for i, line in enumerate(f.xreadlines()):
            if i % 1000 == 0:
                print 'Read file lines:', i
            if i > 0:
                tokens = line.strip().split(',')
                token_2_num[tokens[0][1:-1]] = i - 1
                for doc_num, val in enumerate(tokens[1:]):
                    value = int(val)
                    if value > 0:
                        documents[doc_num].append((i - 1, value))
    num_2_token = {
        v: k
        for k, v in token_2_num.iteritems()
    }

    for doc_num, words in documents.iteritems():
        if doc_num % 100 == 0:
            print 'Processed documents:', doc_num
        
        cnt = Counter()
        cnt_test = Counter()
        
        for word_num, number in words:
            for _ in xrange(number):
                if train_test_split is None or random_gen.random() < train_test_split:
                    cnt[word_num] += 1
                else:
                    cnt_test[word_num] += 1

        if len(cnt) > 0 and (train_test_split is None or len(cnt_test) > 0):
            for w, c in cnt.iteritems():
                row.append(not_empty_docs_number)
                col.append(w)
                data.append(c)
                
            for w, c in cnt_test.iteritems():
                row_test.append(not_empty_docs_number)
                col_test.append(w)
                data_test.append(c)
                
            not_empty_docs_number += 1
            
            if calc_cooccurences:
                keys = [x for x, _ in words]
                doc_cooccurences.update({(w1, w2) for w1 in keys for w2 in keys if w1 != w2})
                doc_occurences.update(keys)

    if train_test_split is None:
        if calc_cooccurences:
            return scipy.sparse.csr_matrix((data, (row, col))), token_2_num, num_2_token, doc_occurences, doc_cooccurences
        else:
            return scipy.sparse.csr_matrix((data, (row, col))), token_2_num, num_2_token
    else:
        if calc_cooccurences:
            return (
                scipy.sparse.csr_matrix((data, (row, col))),
                scipy.sparse.csr_matrix((data_test, (row_test, col_test))),
                token_2_num,
                num_2_token,
                doc_occurences,
                doc_cooccurences
            )
        else:
            return (
                scipy.sparse.csr_matrix((data, (row, col))),
                scipy.sparse.csr_matrix((data_test, (row_test, col_test))),
                token_2_num,
                num_2_token
            )


def create_calculate_likelihood_like_function(n_dw_matrix, loss_function=LogFunction()):
    D, W = n_dw_matrix.shape
    docptr = []
    indptr = n_dw_matrix.indptr
    for doc_num in xrange(D):
        docptr.extend([doc_num] * (indptr[doc_num + 1] - indptr[doc_num]))
    docptr = np.array(docptr)
    wordptr = n_dw_matrix.indices
    
    def fun(phi_matrix, theta_matrix):
        s_data = loss_function.calc(inner1d(theta_matrix[docptr, :], np.transpose(phi_matrix)[wordptr, :]))
        return np.sum(n_dw_matrix.data * s_data)

    return fun


def em_optimization(
    n_dw_matrix, 
    phi_matrix,
    theta_matrix,
    regularization_list,
    iters_count=100,
    loss_function=LogFunction(),
    iteration_callback=None,
    const_phi=False
):
    D, W = n_dw_matrix.shape
    T = phi_matrix.shape[0]
    phi_matrix = np.copy(phi_matrix)
    theta_matrix = np.copy(theta_matrix)
    docptr = []
    indptr = n_dw_matrix.indptr
    for doc_num in xrange(D):
        docptr.extend([doc_num] * (indptr[doc_num + 1] - indptr[doc_num]))
    docptr = np.array(docptr)
    wordptr = n_dw_matrix.indices
    
    start_time = time.time()
    for it in xrange(iters_count):
        phi_matrix_tr = np.transpose(phi_matrix)
        # следующая строчка это 60% времени работы алгоритма
        s_data = loss_function.calc_der(inner1d(theta_matrix[docptr, :], phi_matrix_tr[wordptr, :]))
        # следующая часть это 25% времени работы алгоритма
        A = scipy.sparse.csr_matrix(
            (
                n_dw_matrix.data * s_data, 
                n_dw_matrix.indices, 
                n_dw_matrix.indptr
            ), 
            shape=n_dw_matrix.shape
        )
        A_tr = A.tocsc().transpose()
        # Остальное это 15% времени
        n_tw = np.transpose(A_tr.dot(theta_matrix)) * phi_matrix
        n_dt = A.dot(phi_matrix_tr) * theta_matrix
        
        r_tw, r_dt = regularization_list[it](n_tw, n_dt)
        n_tw += r_tw
        n_dt += n_dt
        n_tw[n_tw < 0] = 0
        n_dt[n_dt < 0] = 0
        
        if not const_phi:
            phi_matrix = n_tw / np.sum(n_tw, axis=1)[:, np.newaxis]
        theta_matrix = n_dt / np.sum(n_dt, axis=1)[:, np.newaxis]
        
        if iteration_callback is not None:
            iteration_callback(it, phi_matrix, theta_matrix)
    
    print 'Iters time', time.time() - start_time
    return phi_matrix, theta_matrix


def naive_thetaless_em_optimization(
    n_dw_matrix, 
    phi_matrix,
    regularization_list,
    iters_count=100,
    iteration_callback=None,
    theta_matrix=None
):
    D, W = n_dw_matrix.shape
    T = phi_matrix.shape[0]
    phi_matrix = np.copy(phi_matrix)
    docptr = []
    indptr = n_dw_matrix.indptr
    for doc_num in xrange(D):
        docptr.extend([doc_num] * (indptr[doc_num + 1] - indptr[doc_num]))
    docptr = np.array(docptr)
    wordptr = n_dw_matrix.indices
    
    start_time = time.time()
    for it in xrange(iters_count):
        phi_rev_matrix = np.transpose(phi_matrix / np.sum(phi_matrix, axis=0))
        theta_matrix = n_dw_matrix.dot(phi_rev_matrix)
        theta_matrix /= np.sum(theta_matrix, axis=1)[:, np.newaxis]
        phi_matrix_tr = np.transpose(phi_matrix)
        
        s_data = 1. / inner1d(theta_matrix[docptr, :], phi_matrix_tr[wordptr, :])
        A = scipy.sparse.csr_matrix(
            (
                n_dw_matrix.data  * s_data , 
                n_dw_matrix.indices, 
                n_dw_matrix.indptr
            ), 
            shape=n_dw_matrix.shape
        ).tocsc()
            
        n_tw = (A.T.dot(theta_matrix)).T * phi_matrix
        r_tw, _ = regularization_list[it](n_tw, theta_matrix)
        n_tw += r_tw
        n_tw[n_tw < 0] = 0
        phi_matrix = n_tw / np.sum(n_tw, axis=1)[:, np.newaxis]

        if iteration_callback is not None:
            iteration_callback(it, phi_matrix, theta_matrix)
    
    print 'Iters time', time.time() - start_time    
    return phi_matrix, theta_matrix


def artm_thetaless_em_optimization(
    n_dw_matrix, 
    phi_matrix,
    regularization_list,
    iters_count=100,
    iteration_callback=None,
    theta_matrix=None
):
    D, W = n_dw_matrix.shape
    T = phi_matrix.shape[0]
    phi_matrix = np.copy(phi_matrix)
    docptr = []
    docsizes = []
    indptr = n_dw_matrix.indptr
    for doc_num in xrange(D):
        size = indptr[doc_num + 1] - indptr[doc_num]
        docptr.extend([doc_num] * size)
        docsizes.extend([size] * size)
    docptr = np.array(docptr)
    wordptr = n_dw_matrix.indices
    docsizes = np.array(docsizes)
    
    B = scipy.sparse.csr_matrix(
        (
            1. * n_dw_matrix.data  / docsizes, 
            n_dw_matrix.indices, 
            n_dw_matrix.indptr
        ), 
        shape=n_dw_matrix.shape
    ).tocsc()
    
    start_time = time.time()
    for it in xrange(iters_count):
        word_norm = np.sum(phi_matrix, axis=0)
        word_norm[word_norm == 0] = 1e-20
        phi_rev_matrix = np.transpose(phi_matrix / word_norm)
        
        theta_matrix = n_dw_matrix.dot(phi_rev_matrix)
        theta_matrix /= np.sum(theta_matrix, axis=1)[:, np.newaxis]
        phi_matrix_tr = np.transpose(phi_matrix)
        
        s_data = 1. / (inner1d(theta_matrix[docptr, :], phi_matrix_tr[wordptr, :]) + 1e-20)
        A = scipy.sparse.csr_matrix(
            (
                n_dw_matrix.data  * s_data , 
                n_dw_matrix.indices, 
                n_dw_matrix.indptr
            ), 
            shape=n_dw_matrix.shape
        ).tocsc()
            
        n_tw = A.T.dot(theta_matrix).T * phi_matrix
        
        r_tw, r_dt = regularization_list[it](n_tw, theta_matrix)
        theta_indices = theta_matrix > 0
        r_dt[theta_indices] /= theta_matrix[theta_indices]
        
        g_dt = A.dot(phi_matrix_tr) + r_dt
        tmp = g_dt.T * B / word_norm
        r_tw += (tmp - np.einsum('ij,ji->i', phi_rev_matrix, tmp)) * phi_matrix
        
        n_tw += r_tw
        n_tw[n_tw < 0] = 0
        phi_matrix = n_tw / np.sum(n_tw, axis=1)[:, np.newaxis]
        phi_matrix[np.isnan(phi_matrix)] = 0.

        if iteration_callback is not None:
            iteration_callback(it, phi_matrix, theta_matrix)
    
    print 'Iters time', time.time() - start_time    
    return phi_matrix, theta_matrix


def gradient_optimization(
    n_dw_matrix, 
    phi_matrix,
    theta_matrix,
    regularization_gradient_list,
    iters_count=100,
    loss_function=LogFunction(),
    iteration_callback=None,
    learning_rate=1.
):
    D, W = n_dw_matrix.shape
    T = phi_matrix.shape[0]
    phi_matrix = np.copy(phi_matrix)
    theta_matrix = np.copy(theta_matrix)
    docptr = []
    indptr = n_dw_matrix.indptr
    for doc_num in xrange(D):
        docptr.extend([doc_num] * (indptr[doc_num + 1] - indptr[doc_num]))
    docptr = np.array(docptr)
    wordptr = n_dw_matrix.indices
    
    start_time = time.time()
    for it in xrange(iters_count):
        phi_matrix_tr = np.transpose(phi_matrix)
        # следующая строчка это 60% времени работы алгоритма
        s_data = loss_function.calc_der(inner1d(theta_matrix[docptr, :], phi_matrix_tr[wordptr, :]))
        # следующая часть это 25% времени работы алгоритма
        A = scipy.sparse.csr_matrix(
            (
                n_dw_matrix.data * s_data, 
                n_dw_matrix.indices, 
                n_dw_matrix.indptr
            ), 
            shape=n_dw_matrix.shape
        ).tocsc()
        # Остальное это 15% времени
        g_tw = theta_matrix.T * A
        g_dt = A.dot(phi_matrix_tr)
        
        r_tw, r_dt = regularization_gradient_list[it](phi_matrix, theta_matrix)
        g_tw += r_tw
        g_dt += r_dt
        
        g_tw -= np.sum(g_tw * phi_matrix, axis=1)[:, np.newaxis]
        g_dt -= np.sum(g_dt * theta_matrix, axis=1)[:, np.newaxis]
        
        phi_matrix += g_tw * learning_rate
        theta_matrix += g_dt * learning_rate
        
        phi_matrix[phi_matrix < 0] = 0
        theta_matrix[theta_matrix < 0] = 0
        
        phi_matrix /= np.sum(phi_matrix, axis=1)[:, np.newaxis]
        theta_matrix /= np.sum(theta_matrix, axis=1)[:, np.newaxis]
        
        if iteration_callback is not None:
            iteration_callback(it, phi_matrix, theta_matrix)
    
    print 'Iters time', time.time() - start_time  
    return phi_matrix, theta_matrix


def svm_score(theta, targets):
    C_2d_range = [1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]
    gamma_2d_range = [1e-3, 1e-2, 1e-1, 1, 1e1]
    best_C, best_gamma, best_val = None, None, 0.
    best_cv_algo_score_on_test = 0.
    X_train, X_test, y_train, y_test = train_test_split(theta, targets, test_size=0.30, stratify=targets, random_state=42)
    for C in C_2d_range:
        for gamma in gamma_2d_range:
            val = np.mean(cross_val_score(SVC(C=C, gamma=gamma), X_train, y_train, scoring='accuracy', cv=4))
            algo = SVC(C=C, gamma=gamma).fit(X_train, y_train)
            test_score = accuracy_score(y_test, algo.predict(X_test))
            print 'SVM(C={}, gamma={}) cv-score: {}  test-score: {}'.format(
                C,
                gamma,
                round(val, 3),
                round(test_score, 3)
            )
            if val > best_val:
                best_val = val
                best_C = C
                best_gamma = gamma
                best_cv_algo_score_on_test = test_score
    print '\n\n\nBest cv params: C={}, gamma={}\nCV score: {}\nTest score:{}'.format(
        best_C,
        best_gamma,
        round(best_val, 3),
        round(best_cv_algo_score_on_test, 3)
    )
    return best_C, best_gamma, best_val, best_cv_algo_score_on_test


def artm_calc_topic_correlation(phi):
    T, W = phi.shape
    return (np.sum(np.sum(phi, axis=0) ** 2) - np.sum(phi ** 2)) / (T * (T - 1))


def artm_get_kernels(phi):
    T, W = phi.shape
    return [
        set(np.where(phi[t, :] * W > 1)[0])
        for t in xrange(T)
    ]


def artm_get_kernels_sizes(phi):
    return [len(kernel) for kernel in artm_get_kernels(phi)]


def artm_get_avg_pairwise_kernels_jacards(phi):
    T, W = phi.shape
    kernels = artm_get_kernels(phi)
    res = 0.
    for i in xrange(T):
        for j in xrange(T):
            if i != j:
                res += 1. * len(kernels[i] & kernels[j]) / len(kernels[i] | kernels[j])
    return res / T / (T - 1)


def artm_calc_perplexity_factory(n_dw_matrix):
    helper = create_calculate_likelihood_like_function(
        loss_function=LogFunction(),
        n_dw_matrix=n_dw_matrix
    )
    total_words_number = n_dw_matrix.sum()
    return lambda phi, theta: np.exp(- helper(phi, theta) / total_words_number)     


def artm_calc_pmi_top_factory(doc_occurences, doc_cooccurences, documents_number, top_size):
    def fun(phi):
        T, W = phi.shape
        pmi = 0.
        for t in xrange(T):
            top = heapq.nlargest(top_size, xrange(W), key=lambda w: phi[t, w])
            for w1 in top:
                for w2 in top:
                    if w1 != w2:
                        pmi += np.log(documents_number * (doc_cooccurences.get((w1, w2), 0.) + 0.1) * 1. / doc_occurences.get(w1, 0) / doc_occurences.get(w2))
        return pmi / (T * top_size * (top_size - 1))
    return fun

