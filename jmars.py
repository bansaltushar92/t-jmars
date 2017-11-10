import logging
from constants import *
from optimize import optimizer
from sampler import GibbsSampler, predicted_rating
import numpy as np
import scipy as sp
from scipy.optimize import fmin_l_bfgs_b
from numpy import linalg as LA
import numpy.matlib
from indexer import Indexer

# Constants
MAX_ITER = 2
MAX_OPT_ITER = 5

def main():
    """
    Main function
    """
    # Download data for NLTK if not already done
    #nltk.download('all')

    # Read 
    imdb = Indexer()
    imdb_file = 'data/clothing_data.json'
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    logging.info('Reading file %s' % imdb_file)
    imdb.read_file(imdb_file)
    logging.info('File %s read' % imdb_file)
    (vocab_size, user_list, movie_list, rating_matrix, review_matrix, review_map, user_dict, movie_dict, rating_list, t_mean) = imdb.get_mappings()
    

    # Get number of users and movies
    Users = len(user_list)
    Movies = len(movie_list)
    logging.info('No. of users U = %d' % Users)
    logging.info('No. of movies M = %d' % Movies)

    # Run Gibbs EM
    for it in range(1,MAX_ITER+1):
        logging.info('Running iteration %d of Gibbs EM' % it)
        logging.info('Running E-Step - Gibbs Sampling')
        gibbs_sampler = GibbsSampler(5,A,2)
        Nums,Numas,Numa = gibbs_sampler.run(rating_matrix, review_map, user_dict, movie_dict)
        #Nums = np.zeros((R,2))
        #Numas = np.zeros((R,A,2))
        #Numa = np.zeros((R,A))
        logging.info('Running M-Step - Gradient Descent')
        for i in range(1,MAX_OPT_ITER+1):
            optimizer(Nums,Numas,Numa,rating_list,t_mean)

if __name__ == "__main__":
    main()