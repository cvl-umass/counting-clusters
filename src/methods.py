import numpy as np
from random import choices
from scipy.spatial.distance import pdist, squareform


def softmax_t(x, tau):
    """ Returns softmax probabilities with temperature tau
        Input:  x -- 1-dimensional array
        Output: s -- 1-dimensional array
    """
    e_x = np.exp(x / tau)
    return e_x / e_x.sum()

def calculate_similarity(data, metric='cisne', tau=0.5):
    '''
    Calculate pair-wise similarity of all elements in dataset. It is produced between each sample features using
    1 - dist, where dist can be cosine, euclidean, or any distance.

    input:
        - data ([lists]):    List with a list per category. Each category list is a list with the elements
                             in each category. Size: [C, (N, D)]
        - metric (string):   Distance metric as in scipy.spatial.distance.pdist's "metric".
                             The distance function can be ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’,
                             ‘correlation’, ‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’, ‘jensenshannon’,
                             ‘kulczynski1’, ‘mahalanobis’, ‘matching’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’,
                             ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’.

    output:
        - s_ij ([lists]):    list of lists with s_ij values. s_ij is the similarity metric between
                             each of the N elements in the dataset. Size: (N,N).
                             It is produced between each sample features using 1 -dist, where dist can be cosine,
                             euclidean, or any distance.
    '''
    data_all = []
    for datai in data:
        for i in datai:
            data_all.append(i)
    if metric == 'minkowski':
        dists = pdist(data_all, metric=metric, p=3)
    else:
        dists = pdist(data_all, metric=metric)
    dists = 1 - dists
    s_ij = squareform(dists) # convert to matrix
    s_ij[s_ij == 0] = 1 # add 1s to the matrix diagonal
    s_ij_sm = np.zeros_like(s_ij)
    for i, s_ij_i in enumerate(s_ij):
        s_ij_sm[i, :] = softmax_t(s_ij_i, tau)

    return s_ij_sm

def nested_is(gt_s_ij, s_ij, N_v, N_n, n_hat=None, ci=False):
    '''
    Function to estimate total number of categories in a dataset using nested importance sampling.

    input:
        - gt_s_ij ([lists]): list of lists with ground truth s_ij values. s_ij is 1 if elements belong to same
                             category, 0 otherwise. Size: (N,N)
        - s_ij ([lists]):    list of lists with s_ij values. s_ij is the similarity metric between
                             each of the N elements in the dataset. Size: (N,N).
                             It is produced between each sample features using 1 -dist, where dist can be cosine,
                             euclidean, or any distance.
        - N_v (int):         Number of sampled vertices
        - N_n (int):         Number of sampled neighbors per vertex
        - n_hat (list):      List with n_hat values (can be calculated only once to save time)
        - ci (bool):         True to return confidence intervals

    output:
        - f_hat (float):     total category count estimate.
        - ci (float):        95% confidence intervals
    '''
    OBSERVATIONS = len(gt_s_ij)

    if not n_hat: # have to be computed only once (saves time)
        # Compute \hat{n}_i
        n_hat = []
        for i in range(OBSERVATIONS):
            n_hat.append(np.sum(s_ij[i]))

    # Proposal distribution to sample a set of n vertices v_i
    Q = (1/np.array(n_hat))/np.sum((1/np.array(n_hat)))
    sampled_vertices = list(np.random.choice(list(range(OBSERVATIONS)), N_v, p=Q, replace=True))

    sampled_neighbors = [] # sampled neighbors per sampled vertex
    q_all = []
    for v_i in sampled_vertices:
        q = (s_ij[v_i])/np.sum((s_ij[v_i])) # proposal distribution to sample a set of neighbors for each sampled vertex
        q_all.append(q)
        sampled_neighbors.append([v_i]+list(np.random.choice(list(range(len(s_ij[v_i]))), N_n-1, p=q, replace=True)))

    # Estimate \bar{n}
    sum_cc = 0
    for i, v_i in enumerate(sampled_vertices):
        sum_n_bar = 0
        for j, v_j in enumerate(sampled_neighbors[i]):
            sum_n_bar += gt_s_ij[v_i][v_j]/q_all[i][v_j]
        n_bar = sum_n_bar/len(sampled_neighbors[i])
        sum_cc += (1/n_bar)*(1/Q[v_i])
    f_hat = sum_cc/len(sampled_vertices)

    if not ci:
        return f_hat, n_hat
    else:
        # Estimate variance and calculate confidence intervals
        w_ci = 0
        for i, v_i in enumerate(sampled_vertices):
            sum_n_bar = 0
            for v_j in sampled_neighbors[i]:
                sum_n_bar += gt_s_ij[v_i][v_j]/q_all[i][v_j]
            n_bar = sum_n_bar/len(sampled_neighbors[i])
            w_ci += ((1/n_bar)*(1/Q[v_i]) - f_hat)**2
        var_hat = w_ci/N_v # estimated variance
        ci = 1.96*(np.sqrt(var_hat/N_v)) # 95% confidence intervals
        return f_hat, ci, n_hat

def nested_mc(gt_s_ij, N_v, N_n, ci=False):
    '''
    Function to estimate population size in a dataset using simple nested Monte Carlo sampling.
    '''
    OBSERVATIONS = len(gt_s_ij)

    # Proposal distribution to sample a set of n vertices v_i
    sampled_vertices = choices(list(range(OBSERVATIONS)), k=N_v) # sample uniformly random w/ replacement

    sampled_neighbors = [] # sampled neighbors per sampled vertex
    for v_i in sampled_vertices:
        sampled_neighbors.append([v_i]+choices(list(range(len(gt_s_ij[v_i]))), k=N_n-1)) # to guarantee it samples itself

    # Estimate \bar{n}
    sum_cc = 0
    for i, v_i in enumerate(sampled_vertices):
        sum_n_bar = 0
        for v_j in sampled_neighbors[i]:
            sum_n_bar += gt_s_ij[v_i][v_j]
        n_bar = sum_n_bar/len(sampled_neighbors[i])*OBSERVATIONS
        sum_cc += 1/n_bar
    f_hat = sum_cc/len(sampled_vertices)*OBSERVATIONS

    if not ci:
        return f_hat
    else:
        # Estimate variance and calculate confidence intervals
        w_ci = 0
        for i, v_i in enumerate(sampled_vertices):
            sum_n_bar = 0
            for v_j in sampled_neighbors[i]:
                sum_n_bar += gt_s_ij[v_i][v_j]
            n_bar = sum_n_bar/len(sampled_neighbors[i])*OBSERVATIONS
            w_ci += ((1/n_bar)*OBSERVATIONS - f_hat)**2
        var_hat = w_ci/len(sampled_vertices) # estimated variance
        ci = 1.96*(np.sqrt(var_hat/N_v)) # 95% confidence intervals
        return f_hat, ci



