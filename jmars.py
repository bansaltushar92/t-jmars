import logging
from constants import *
from sampler import GibbsSampler
from optimize import optimizer
import numpy as np
import numpy.matlib
from indexer import Indexer

# Constants
MAX_ITER = 20
MAX_OPT_ITER = 10

def main():
    """
    Main function
    """
    # Download data for NLTK if not already done
    #nltk.download('all')

    # Read 
    imdb = Indexer()
    imdb_file = 'data/clothing_data_small.json'
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    logging.info('Reading file %s' % imdb_file)
    imdb.read_file(imdb_file)
    logging.info('File %s read' % imdb_file)
    
    (vocab_size, 
        user_list,  # remove
        movie_list, 
        review_matrix, 
        review_map, 
        user_dict, 
        movie_dict, 
        rating_list, 
        t_mean, 
        movie_reviews, 
        word_dictionary,
        U, M, R) = imdb.get_mappings()
    

    ## Initialize
    alpha_vu = np.random.normal(0,sigma_u,(U, K))
    alpha_bu = np.random.normal(0,sigma_u,(U, 1))
    alpha_tu = np.random.normal(0,sigma_u,(U, A))
    
    
    # User
    v_u = np.random.normal(0,sigma_u,(U, K))      # Latent factor vector
    b_u = np.random.normal(0,sigma_bu,(U, 1))      # Common bias vector
    theta_u = np.random.normal(0,sigma_ua,(U, A))  # Aspect specific vector
    
    # Movie
    v_m = np.random.normal(0,sigma_m,(M, K))      # Latent factor vector
    b_m = np.random.normal(0,sigma_bm,(M, 1))      # Common bias vector
    theta_m = np.random.normal(0,sigma_ma,(M, A))  # Aspect specific vector
    
    # Common bias
    b_o = np.random.normal(0,sigma_b0) 
    
    # Scaling Matrix
    M_a = np.random.normal(0,sigma_Ma,(A, K))
    
    params = numpy.concatenate((alpha_vu.flatten('F'), 
                                    v_u.flatten('F'), 
                                    alpha_bu.flatten('F'), 
                                    b_u.flatten('F'), 
                                    alpha_tu.flatten('F'), 
                                    theta_u.flatten('F'), 
                                    v_m.flatten('F'), 
                                    b_m.flatten('F'), 
                                    theta_m.flatten('F'), 
                                    M_a.flatten('F'), 
                                    np.array([b_o]).flatten('F')))
    # Get number of users and movies
    Users = len(user_list)
    Movies = len(movie_list)
    logging.info('No. of users U = %d' % Users)
    logging.info('No. of movies M = %d' % Movies)


    # change gibbs sampler initialization
    gibbs_sampler = GibbsSampler(vocab_size,
                                    review_matrix,
                                    rating_list,
                                    movie_dict,
                                    user_dict,
                                    movie_reviews,
                                    word_dictionary
                                    ,U, M, R)


    # Run Gibbs EM
    for it in range(1,MAX_ITER+1):
        print('Running iteration %d of Gibbs EM' % it)
        print('Running E-Step - Gibbs Sampling')

        Nums,Numas,Numa = gibbs_sampler.run(vocab_size, 
                                            review_matrix, 
                                            rating_list, 
                                            user_dict, 
                                            movie_dict, 
                                            movie_reviews, 
                                            word_dictionary, 
                                            t_mean, 
                                            params)
#        Nums = np.zeros((R,2))
#        Numas = np.zeros((R,A,2))
#        Numa = np.zeros((R,A))
        print('Running M-Step - Gradient Descent')
        for i in range(1,MAX_OPT_ITER+1):
            params = optimizer(Nums,Numas,Numa,rating_list,t_mean,params,U,M,R)
    
if __name__ == "__main__":
    main()