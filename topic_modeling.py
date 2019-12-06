'''
Data Mining Project

Print topics found in Yelp business reviews

If training is specified, dataset will be read from disk and model will be trained.
Otherwise, pickled model will be read from disk.
'''

import argparse
from collections import defaultdict
import json
import pickle
import os
import sys
from time import time

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import nltk
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(400)


PICKLED_MODEL_FILENAME = 'lda_model.p'
PICKLED_RESTAURANT_REVIEW_DATA_FILENAME = 'restaurants.p'

default_dataset_path = os.path.join(
	os.path.dirname(__file__),
	'dataset',
)

parser = argparse.ArgumentParser()

parser.add_argument(
	'-t', '--train',
	action='store_true',
	help='train new model (Yelp dataset must also be present on disk))',
)

parser.add_argument(
	'-d', '--dataset',
	default=default_dataset_path,
	help='specify absolute path to dataset',
)

parser.add_argument(
	'-n', '--num_topics',
	default=14,
	help='number of topics to train LDA on',
)

# Write functions to perform the preprocessing
stemmer = SnowballStemmer('english')
def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


# Tokenize and lemmatize
def preprocess(text):
    result=[]
    for token in simple_preprocess(text, deacc=True, min_len=4):
        if token not in STOPWORDS:
            result.append(lemmatize_stemming(token))
    return result




if __name__ == '__main__':
	args = parser.parse_args()
	arg_dict = vars(args)

	if arg_dict['train']:

		# try to grab pickled version of dataset, if possible
		if not os.path.exists(PICKLED_RESTAURANT_REVIEW_DATA_FILENAME):
			print('importing dataset...')

			dataset_path = arg_dict['dataset']

			try:
				datasets = {
					f.split('.')[0]: dataset_path + os.sep + f for f in filter(lambda x: x.endswith('.json'), os.listdir(dataset_path))
				}
			except:
				print('Could not find dataset at {}'.format(dataset_path))
				sys.exit(0)

			# organize data to be used in training model
			restaurants = defaultdict(dict)

			with open(datasets['business']) as f:
				for line in f:
					contents = json.loads(line)
					if contents['categories'] and 'Restaurants' in contents['categories']:
						restaurants[contents['business_id']]['categories'] = contents['categories']
						restaurants[contents['business_id']]['reviews'] = []

			with open(datasets['review']) as f:
				for line in f:
					contents = json.loads(line)
					if contents['business_id'] and contents['business_id'] in restaurants:
						restaurants[contents['business_id']]['reviews'].append(contents['text'])

			all_the_reviews = []
			for v in restaurants.values():
				all_the_reviews += v['reviews']

			print('Saving pickled dataset to disk as {}'.format(PICKLED_RESTAURANT_REVIEW_DATA_FILENAME))
			pickle.dump(all_the_reviews, open(PICKLED_RESTAURANT_REVIEW_DATA_FILENAME, 'wb'))

		else:
			print('Using pickled dataset file found on disk as {}'.format(PICKLED_RESTAURANT_REVIEW_DATA_FILENAME))
			all_the_reviews = pickle.load(open(PICKLED_RESTAURANT_REVIEW_DATA_FILENAME, 'rb'))

		# grab wordnet if not present for preprocessing data
		nltk.download('wordnet')

		print('Preview of document after preprocessing')

		doc_sample = all_the_reviews[0]

		print("Original document: ")
		words = [word for word in doc_sample.split(' ')]
		print("\nBefore preprocessing: ")
		print(words)
		print("\nTokenized and lemmatized document: ")
		print(preprocess(doc_sample))

		# Need to split reviews into training and testing
		training_reviews = all_the_reviews[:50000]
		processed_reviews = [preprocess(review) for review in training_reviews]

		'''
		Bag of words on the dataset
		Create a dictionary from 'processed_docs' containing the number of times a word appears
		in the training set. To do that, we pass processed_docs to gensim.corpora.Dictionary()
		'''
		dictionary = gensim.corpora.Dictionary(processed_reviews)

		'''
		Filter out tokens that appear only a few times or in more than 10% of documents
		'''
		dictionary.filter_extremes(
			no_below=0.001*len(processed_reviews),
			no_above=0.1,
		)

		'''
		Create the Bag-of-words model for each document i.e for each document we create a dictionary reporting how many
		words and how many times those words appear. Save this to 'bow_corpus'
		'''
		bow_corpus = [dictionary.doc2bow(review) for review in processed_reviews]

		'''
		Preview BOW for our sample preprocessed document
		'''
		print('\nBag of Words Preview on example document:')
		print(training_reviews[25])

		for doc in bow_corpus[25]:
		    print("Word {} (\"{}\") appears {} time.".format(doc[0], dictionary[doc[0]], doc[1]))

		'''
		Running LDA using Bag of Words
		We are going for 10 topics in the document corpus.

		We will be running LDA using 8 CPU cores to parallelize and speed up model training.
		'''
		num_topics = int(arg_dict['num_topics'])
		print('\nBuilding LDA model with {} topics...'.format(num_topics))
		lda_model =  gensim.models.LdaMulticore(
			bow_corpus,
			num_topics=num_topics,
			id2word=dictionary,
			passes=2,
			workers=8,
		)

		# serialize and save LDA model object to disk
		if not os.path.exists(PICKLED_MODEL_FILENAME):
			print('\nsaving pickled model to disk {}'.format(PICKLED_MODEL_FILENAME))
			pickle.dump(lda_model, open(PICKLED_MODEL_FILENAME, 'wb'))

	else:
		try:
			lda_model = pickle.load(open(PICKLED_MODEL_FILENAME, 'rb'))
		except:
			print('\nPickled LDA model not found on disk as {}... try running program with -t option to train new model!'.format(PICKLED_MODEL_FILENAME))
			sys.exit(0)

	# print discovered topics to screen
	for i, topic in lda_model.print_topics(
		num_topics=-1,
		num_words=20,
	):
	    print("Topic: {} \nWords: {}".format(i, topic))
