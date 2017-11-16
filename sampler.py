from constants import *
import numpy as np
from scipy.sparse import dok_matrix


def assignment(x):

    global alpha_vu
    global v_u
    
    global alpha_bu
    global b_u
    
    global alpha_tu
    global theta_u

    global v_m
    global b_m
    global theta_m

    global M_a
    global b_o

    alpha_vu = x[:(U*K)].reshape((U,K), order='F')
    v_u = x[(U*K):2*(U*K)].reshape((U,K), order='F')
    
    alpha_bu = x[2*(U*K):2*(U*K) + U].reshape((U,1), order='F')
    b_u = x[2*(U*K) + U:2*(U*K) + 2*U].reshape((U,1), order='F')
    
    alpha_tu = x[2*(U*K) + 2*U:(2*(U*K) + 2*U + U*A)].reshape((U,A), order='F')
    theta_u = x[(2*U*K + 2*U + U*A): (2*U*K + 2*U + 2*U*A)].reshape((U,A), order='F')

    v_m = x[((2*U*K + 2*U + 2*U*A)):((2*U*K + 2*U + 2*U*A) + M*K)].reshape((M,K), order='F')
    b_m = x[((2*U*K + 2*U + 2*U*A)+ M*K):((2*U*K + 2*U + 2*U*A) + M*K + M)].reshape((M,1), order='F')
    theta_m = x[((2*U*K + 2*U + 2*U*A) + M*K + M):((2*U*K + 2*U + 2*U*A) + M*K + M + M*A)].reshape((M,A), order='F')

    M_a = x[((2*U*K + 2*U + 2*U*A) + M*K + M + M*A):((2*U*K + 2*U + 2*U*A) + M*K + M + M*A + A*K)].reshape((A,K), order='F')
    b_o = x[-1]


def dev_t(t, tu_mean):
    # return np.sign(t-tu_mean)*abs(t-tu_mean)**beta
    return 0.0

# Joint aspect distribution
def joint_aspect(u, m, t, t_mean_u):
    """
    Returns the joint aspect distribution
    """
    
    num_theta_uma = np.exp(theta_u[u] + dev_t(t, t_mean_u)*alpha_tu[u] + theta_m[m])
    theta_uma = np.divide(num_theta_uma.T,num_theta_uma.sum()).T

    return theta_uma

def predicted_rating(u, m, t, t_mean_u):
    """
    Computes the predicted rating for user u on movie m
    """

    theta_uma = joint_aspect(u, m, t, t_mean_u)
    M_sum = np.dot(theta_uma, M_a)

    v_ut = v_u[u] + dev_t(t, t_mean_u)*alpha_vu[u]
    b_ut = b_u[u] + dev_t(t, t_mean_u)*alpha_bu[u]

    r = np.dot(np.dot(v_ut, np.diag(M_sum)), v_m[m].T) + b_o + b_ut + b_m[m]  
    return r

def predicted_aspect_rating(u, m, a, t, t_mean_u):
    """
    Computes the predicted rating for user u on movie m and aspect a
    """
    v_ut = v_u[u] + dev_t(t, t_mean_u)*alpha_vu[u]
    b_ut = b_u[u] + dev_t(t, t_mean_u)*alpha_bu[u]
    
    r = np.dot(np.dot(v_ut, np.diag(M_a[a])), v_m[m].T) + b_o + b_ut + b_m[m]
    return r

def aspect_sentiment_probability(s, u, m, a, t, t_mean_u):
    """
    Computes the probability for a sentiment s on aspect a 
    for user u on movie m
    """
    ruma = predicted_aspect_rating(u,m,a,t, t_mean_u)
    prob_suma = 1.0 / (1.0 + np.exp(-s*(c*ruma - b)))
    return prob_suma

def aggregate_sentiment_probability(s, u, m, t, t_mean_u):
    """
    Computes the probability for aggregate sentiment s 
    for user u and movie m
    """
    rum = predicted_rating(u,m,t, t_mean_u)
    prob_sum = 1.0 / (1.0 + np.exp(-s*(c*rum - b)))
    return prob_sum

def sample_multinomial(w):
    """
    Returns the index of a sample from a multinomial distribution
    """
    x = np.random.uniform(0,1)
    for i,v in enumerate(np.cumsum(w)):
        if x < v: return i
    return len(w)-1

def sample_multiple_indices(p):
    """
    Samples indices from a joint probability distribution
    """
    (Y, Z, S) = p.shape
    dist = list()
    for y in range(Y):
        for z in range(Z):
            for s in range(S):
                dist.append(p[y,z,s])
    index = sample_multinomial(dist)
    y = index // (Z * S)                 #Get indices of matrix from the list indices.
    rem = index % (Z * S)
    z = rem // S
    s = rem % S
    #print(y, z, s)
    return (y, z, s)

def word_indices(vec):
    """
    Returns non-zero entries of vec one at a time
    """
#    for idx in vec.nonzero()[0]:
#        for i in range(int(vec[idx])):
#            yield idx

class GibbsSampler:
    """
    Class to handle Gibbs Sampling
    """
    def __init__(self, Y, Z, S):
        """
        Constructor
        """
        self.Y = Y
        self.Z = Z
        self.S = S
        self.M = M
        self.U = U
        self.R = R
        self.A = A

    def _initialize(self, vocab_size, review_matrix, rating_list, movie_dict, user_dict, movie_reviews, word_dictionary):

        """
        Initialize all variables needed in the run step
        """
        self.vocab_size = vocab_size
        self.n_reviews = len(review_matrix)

        # Number of times y occurs
        self.cy = np.zeros(self.Y)
        self.c = 0
        # Number of times y occurs with w
        self.cyw = np.zeros((self.Y, self.vocab_size))
        # Number of times y occurs with s and w
        self.cysw = np.zeros((self.Y, self.S, self.vocab_size))
        # Number of times y occurs with s
        self.cys = np.zeros((self.Y, self.S))
        # Number of times y occurs with z and w
        self.cyzw = np.zeros((self.Y, self.Z, self.vocab_size))
        # Number of times y occurs with z
        self.cyz = np.zeros((self.Y, self.Z))

        
        # Number of times y occurs with m and w
#        self.cymw = dok_matrix((self.Y, self.M, self.vocab_size), dtype=np.float)
        # Number of times y occurs with m
#        self.cym = np.zeros((self.Y, self.M))

        self.Nums = np.zeros((self.R,2))
        self.Numas = np.zeros((self.R,self.A,2))
        self.Numa = np.zeros((self.R,self.A))

        self.topics = {}

        for movie in range(self.M):
            for (rev,r) in movie_reviews[movie]:
                for i, word in enumerate(rev.strip().split()):
                    w = word_dictionary[word]
                    # Choose a random assignment of y, z, w
                    (y, z, s) = (np.random.randint(self.Y), np.random.randint(self.Z), np.random.randint(self.S))
                    # Assign new values
                    self.cy[y] += 1
                    self.c += 1
                    self.cyw[y,w] += 1
                    self.cy[y] += 1
                    self.cysw[y,s,w] += 1
                    self.cys[y,s] += 1
                    self.cyzw[y,z,w] += 1
                    self.cyz[y,z] += 1
                    
                    #m = np.random.randint(self.M)
                    #self.cymw[y,m,w] += 1
                    #self.cym[y,m] += 1
    
                    # Get Movie and User
    
                    #m = movie_dict[review_map[r]['movie']]
                    #u = user_dict[review_map[r]['user']]
    
                     # TODO update nums
    
                    self.Nums[r,s] +=1
                    self.Numas[r,z,s] +=1
                    self.Numa[r,z] +=1
    
                    self.topics[(r, i)] = (y, z, s)
            #print(list((review_matrix[self.n_reviews-1, :])))

    def _conditional_distribution(self, u, w, m, t, t_mean_u, cymw, cym):
        """
        Returns the CPD for word w in the review by user u for movie m
        """
        p_z = np.zeros((self.Y, self.Z, self.S))
        # y = 0
        for z in range(self.Z):
            for s in range(self.S):
                p_z[0,z,s] = (self.cy[0] + gamma) / (self.c + 5 * gamma)
                p_z[0,z,s] = (p_z[0,z,s] * (self.cyw[0,w] + eta)) / (self.cy[0] + eta)

        # y = 1
        for z in range(self.Z):
            for s in range(self.S):
                p_z[1,z,s] = (self.cy[1] + gamma) / (self.c + 5 * gamma)
                p_z[1,z,s] = (p_z[1,z,s] * (self.cysw[1,s,w] + eta)) / (self.cys[1,s] + eta)
                p_z[1,z,s] = p_z[1,z,s] * aggregate_sentiment_probability(s,u,m,t, t_mean_u)

        # y = 2
        for z in range(self.Z):
            for s in range(self.S):
                p_z[2,z,s] = (self.cy[2] + gamma) / (self.c + 5 * gamma)
                p_z[2,z,s] = (p_z[2,z,s] * (self.cyzw[2,z,w] + eta)) / (self.cyz[2,z] + eta)
                p_z[2,z,s] = p_z[2,z,s] * (joint_aspect(u, m,t, t_mean_u)[z])
                p_z[2,z,s] = p_z[2,z,s] * aspect_sentiment_probability(s,u,m,z,t, t_mean_u)

        # y = 3
        for z in range(self.Z):
            for s in range(self.S):
                p_z[3,z,s] = (self.cy[3] + gamma) / (self.c + 5 * gamma)
                p_z[3,z,s] = (p_z[3,z,s] * (self.cyzw[3,z,w] + eta)) / (self.cyz[3,z] + eta)
                p_z[3,z,s] = p_z[3,z,s] * (joint_aspect(u,m,t, t_mean_u)[z])

        # y = 4
        for z in range(self.Z):
            for s in range(self.S):
                p_z[4,z,s] = (self.cy[4] + gamma) / (self.c + 5 * gamma)
                p_z[4,z,s] = (p_z[4,z,s] * (cymw[4,w] + eta)) / (cym[4] + eta)

        # Normalize
        p_z = p_z / p_z.sum()

        return p_z

    def run(self, vocab_size, review_matrix, rating_list, user_dict, movie_dict, movie_reviews, word_dictionary,t_mean, params, max_iter=1):
        """
        Perform sampling max_iter times
        """
        self._initialize(vocab_size, review_matrix, rating_list, movie_dict, user_dict, movie_reviews, word_dictionary)
        
        assignment(params)

        #print("hello 3 GIBBS_BIATCH", M_a)
        for it in range(max_iter):
            print("iter-> ", it)
            print('Gibbs Sampling Iteration: %d' % it)
            for movie in range(self.M):
                
                if(movie%1000 == 0):
                    print(movie)
                
                # Number of times y occurs with m and w
                cymw = dok_matrix((self.Y, self.vocab_size), dtype=np.float)
                # Number of times y occurs with m
                cym = np.zeros(self.Y)
                
                for (rev,r) in movie_reviews[movie]:
                    for w in rev.strip().split():       ## map w to it's index
                        (y, z, s) = (np.random.randint(self.Y), np.random.randint(self.Z), np.random.randint(self.S))
                        cymw[y,word_dictionary[w]] += 1
                        cym[y] += 1
                
                
                for (rev,r) in movie_reviews[movie]:
                    for i, word in enumerate(rev.strip().split()):
                        w = word_dictionary[word]
                        (y, z, s) = self.topics[(r, i)]
                        
                        # Exclude current assignment
                        self.cy[y] -= 1 # specific to y
                        self.c -= 1     # sum over all y
                        self.cyw[y,w] -= 1
                        # self.cy[y] -= 1   # Note: Wrong should not be reduced again
                        self.cysw[y,s,w] -= 1
                        self.cys[y,s] -= 1
                        self.cyzw[y,z,w] -= 1
                        self.cyz[y,z] -= 1
                        
                        cymw[y,w] -= 1
                        cym[y] -= 1
    
                        # Get next distribution
                        
                        u = rating_list[r]['u'] #np.random.randint(1000)
                        t = rating_list[r]['t']
                        self.Nums[r,s] -=1
                        self.Numas[r,z,s] -=1
                        self.Numa[r,z] -=1
                        
                        p_z = self._conditional_distribution(u, w, movie, t, t_mean[u], cymw, cym) # Eq. 13 for all values of y,z,s -> computing tensor
                        (y, z, s) = sample_multiple_indices(p_z)
    
                        # Assign new values
                        self.cy[y] += 1
                        self.c += 1
                        self.cyw[y,w] += 1
                        self.cy[y] += 1
                        self.cysw[y,s,w] += 1
                        self.cys[y,s] += 1
                        self.cyzw[y,z,w] += 1
                        self.cyz[y,z] += 1
                        
                        cymw[y,w] += 1
                        cym[y] += 1
                        
                        self.Nums[r,s] +=1
                        self.Numas[r,z,s] +=1
                        self.Numa[r,z] +=1
                        
                        self.topics[(r, i)] = (y, z, s)
            #print(self.Nums[1], self.Numas[1], self.Numa[1])
            return (self.Nums, self.Numas, self.Numa)
        
