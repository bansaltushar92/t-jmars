import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
from collections import defaultdict

def clean_review(review):
    """
    Removes punctuations, stopwords and returns an array of words
    """
    review = review.replace('!', ' ')
    review = review.replace('?', ' ')
    review = review.replace(':', ' ')
    review = review.replace(';', ' ')
    review = review.replace(',', ' ')
    tokens = word_tokenize(review)
    tokens = [w for w in tokens if w not in stopwords.words('english')]
    return tokens

class Indexer:
    """
    Class to load data from file and obtain relevant data structures
    """
    def __init__(self):
        """
        Constructor
        """
        self.reviews = list()

    def read_file(self, filename):
        """
        Reads reviews from a specified file
        """
        f = open(filename)
        data = f.read()
        self.reviews = json.loads(data)

    def get_mappings(self):
        """
        Returns relevant data like vocab size, user list, etc after
        processing review data
        """
        user_dict = dict()
        movie_dict = dict()
        rating_list = []
        t_sum = defaultdict(int)
        t_count = defaultdict(int)
        
        for review in self.reviews:
            user = review['reviewerID'] #['user']
            if user not in user_dict:
                nu = len(user_dict.keys())
                user_dict[user] = nu
                t_sum[user] += review['unixReviewTime']
                t_count[user] += 1
            
            movie = review['asin'] #['movie']
            if movie not in movie_dict:
                nm = len(movie_dict.keys())
                movie_dict[movie] = nm
        
        nu = len(user_dict.keys())
        t_mean = np.zeros(len(user_dict.keys()))
        user_list = [''] * nu
        for user in user_dict:
            idx = user_dict[user]
            user_list[idx] = user
            t_mean[idx] = t_sum[user]/t_count[user]
            
        nm = len(movie_dict.keys())
        movie_list = [''] * nm
        for movie in movie_dict:
            idx = movie_dict[movie]
            movie_list[idx] = movie 
        
        word_dictionary = dict()
        review_matrix = list()
        word_index = 0
#        for review in self.reviews:
#            if type(review['review']) == str:
#                temp = review['review']
#            else:
#                try:
#                    temp = review['review']['review']
#                except:
#                    continue
            #arr = clean_review(temp)

        
        
        review_map = list()
        movie_reviews = [[] for o in range(len(movie_dict))]
        for index in range(len(self.reviews)):
            
            review_map.append(
            {
                'user' : self.reviews[index]['reviewerID'],  #['user'],
                'movie' : self.reviews[index]['asin']  #['movie']
            })
            
            movie_reviews[movie_dict[self.reviews[index]['asin']]].append((self.reviews[index]['reviewText'], index))
            
            rating_list.append({'u': user_dict[self.reviews[index]['reviewerID']], 'm': movie_dict[self.reviews[index]['asin']], 't': self.reviews[index]['unixReviewTime'], 'r':self.reviews[index]['overall']})
            temp = self.reviews[index]['reviewText']
            arr = temp.split()
            review_matrix.append(arr)
            for ar in arr:
                ar = ar.strip()
                if ar not in word_dictionary:
                    word_dictionary[ar] = word_index
                    word_index += 1
        
        vocab_size = len(word_dictionary.keys())
        review_matrix = np.array(review_matrix)
         
        print(len(movie_dict))
        return (vocab_size, user_list, movie_list, review_matrix, review_map, user_dict, movie_dict, rating_list, t_mean, movie_reviews, word_dictionary,nu,nm,len(self.reviews))