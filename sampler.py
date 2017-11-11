from constants import *
import numpy as np

# Joint aspect distribution
def joint_aspect(u, m):
    """
    Returns the joint aspect distribution
    """
    u_a = theta_u[u].T
    m_a = theta_m[m].T
    um_a = np.exp(np.add(u_a, m_a))
    theta_um = um_a / np.sum(um_a)
    return theta_um

def predicted_rating(u, m):
    """
    Computes the predicted rating for user u on movie m
    """
    theta_um = joint_aspect(u, m)
    temp = np.diag((np.dot(M_a.T, theta_um)).reshape(K))
    r = v_u[u].dot(temp).dot(v_m[m].T) + b_o + b_u[u] + b_m[m]
    return r.sum()

def predicted_aspect_rating(u, m, a):
    """
    Computes the predicted rating for user u on movie m and aspect a
    """
    
    temp = np.diag(M_a[a])
    r = v_u[u].dot(temp).dot(v_m[m].T) + b_o + b_u[u] + b_m[m]
    return r.sum()

def aspect_sentiment_probability(s, u, m, a):
    """
    Computes the probability for a sentiment s on aspect a 
    for user u on movie m
    """
    ruma = predicted_aspect_rating(u,m,a)
    prob_suma = 1.0 / (1.0 + np.exp(-s*(c*ruma - b)))
    return prob_suma

def aggregate_sentiment_probability(s, u, m):
    """
    Computes the probability for aggregate sentiment s 
    for user u and movie m
    """
    rum = predicted_rating(u,m)
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
    for idx in vec.nonzero()[0]:
        for i in range(int(vec[idx])):
            yield idx

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

    def _initialize(self, review_matrix, review_map, movie_dict, user_dict):

        """
        Initialize all variables needed in the run step
        """
        (self.n_reviews, self.vocab_size) = review_matrix.shape

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
        self.cymw = np.zeros((self.Y, self.M, self.vocab_size))
        # Number of times y occurs with m
        self.cym = np.zeros((self.Y, self.M))

        self.Nums = np.zeros((self.R,2))
        self.Numas = np.zeros((self.R,self.A,2))
        self.Numa = np.zeros((self.R,self.A))

        self.topics = {}



        for r in range(self.n_reviews):

            for i, w in enumerate(word_indices(review_matrix[r, :])):
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
                # TODO: Define m
                m = np.random.randint(self.M)
                self.cymw[y,m,w] += 1
                self.cym[y,m] += 1

                # Get Movie and User

                #m = movie_dict[review_map[r]['movie']]
                #u = user_dict[review_map[r]['user']]

                 # TODO update nums

                self.Nums[r,s] +=1
                self.Numas[r,z,s] +=1
                self.Numa[r,z] +=1

                self.topics[(r, i)] = (y, z, s)

    def _conditional_distribution(self, u, m, w):
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
                p_z[1,z,s] = p_z[1,z,s] * aggregate_sentiment_probability(s,u,m)

        # y = 2
        for z in range(self.Z):
            for s in range(self.S):
                p_z[2,z,s] = (self.cy[2] + gamma) / (self.c + 5 * gamma)
                p_z[2,z,s] = (p_z[2,z,s] * (self.cyzw[2,z,w] + eta)) / (self.cyz[2,z] + eta)
                p_z[2,z,s] = p_z[2,z,s] * (joint_aspect(u, m)[z])
                p_z[2,z,s] = p_z[2,z,s] * aspect_sentiment_probability(s,u,m,z)

        # y = 3
        for z in range(self.Z):
            for s in range(self.S):
                p_z[3,z,s] = (self.cy[3] + gamma) / (self.c + 5 * gamma)
                p_z[3,z,s] = (p_z[3,z,s] * (self.cyzw[3,z,w] + eta)) / (self.cyz[3,z] + eta)
                p_z[3,z,s] = p_z[3,z,s] * (joint_aspect(u,m)[z])

        # y = 4
        for z in range(self.Z):
            for s in range(self.S):
                p_z[4,z,s] = (self.cy[4] + gamma) / (self.c + 5 * gamma)
                p_z[4,z,s] = (p_z[4,z,s] * (self.cymw[4,m,w] + eta)) / (self.cym[4,m] + eta)

        # Normalize
        p_z = p_z / p_z.sum()

        return p_z

    def run(self, rating_matrix, review_map, user_dict, movie_dict, max_iter=1):
        """
        Perform sampling max_iter times
        """
        self._initialize(rating_matrix, review_map, movie_dict, user_dict)

        print (M_a)
        for it in range(max_iter):
            print("iter-> ", it)
            print('Gibbs Sampling Iteration: %d' % it)
            for r in range(self.n_reviews):
                for i, w in enumerate(word_indices(rating_matrix[r, :])): # iterate over word index for ratings non-zero
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
                    # TODO: Define m
                    m = movie_dict[review_map[r]['movie']] #np.random.randint(self.M)   # Why random ?? ?????? Take specific
                    #print "error", y, m, w
                    self.cymw[y,m,w] -= 1
                    self.cym[y,m] -= 1

                    # Get next distribution
                    # TODO: Define u
                    u = user_dict[review_map[r]['user']] #np.random.randint(1000)   # Why random ?? ?????? Take specific
                    self.Nums[r,s] -=1
                    self.Numas[r,z,s] -=1
                    self.Numa[r,z] -=1
                    
                    p_z = self._conditional_distribution(u, m, w) # Eq. 13 for all values of y,z,s -> computing tensor
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
                    # TODO: Define m
                    m = np.random.randint(self.M)
                    self.cymw[y,m,w] += 1
                    self.cym[y,m] += 1
                    
                    self.Nums[r,s] +=1
                    self.Numas[r,z,s] +=1
                    self.Numa[r,z] +=1
                    
                    self.topics[(r, i)] = (y, z, s)
  
            return (self.Nums, self.Numas, self.Numa)