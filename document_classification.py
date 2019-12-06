'''
Data Mining Project

Predict labels for Yelp business reviews

If training is specified, dataset will be read from disk and model will be trained.
Otherwise, pickled classifier will be read from disk.
'''

import argparse
from collections import defaultdict
import json
import numpy as np
import os
import pickle
import sys
from time import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


PICKLED_CLASSIFIER_FILENAME = 'clf.p'
TEST_TFIDF_FILENAME = 'test_tfidf.p'
TEST_TARGET_FILENAME = 'test_target.p'

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


if __name__ == '__main__':
	args = parser.parse_args()
	arg_dict = vars(args)

	if arg_dict['train']:

		print('importing dataset...')
		dataset_path = arg_dict['dataset']

		try:
			datasets = {
				f.split('.')[0]: dataset_path + os.sep + f for f in filter(lambda x: x.endswith('.json'), os.listdir(dataset_path))
			}
		except:
			print('Could not find dataset at {}'.format(dataset_path))
			sys.exit(0)

		# organize data to be used in training and testing model
		reviews = defaultdict(list)
		with open(datasets['review']) as f:
			for line in f:
				contents = json.loads(line)
				reviews[contents['stars']].append(contents['text'])

		balanced_reviews = []
		balanced_ratings = []
		for k, v in reviews.items():
			for review in v[:2000]:
				balanced_reviews.append(review)
				balanced_ratings.append(k)

		test_reviews = []
		test_ratings = []
		for k, v in reviews.items():
			for review in v[-1000:]:
				test_reviews.append(review)
				test_ratings.append(k)

		# vectorize documents to be converted to bag of words and used for training
		vectorizer = TfidfVectorizer()
		train_tfidf = vectorizer.fit_transform(balanced_reviews)
		train_target = balanced_ratings

		test_tfidf = vectorizer.transform(test_reviews)
		test_target = test_ratings

		# serialize test data and save to disk
		if not (os.path.exists(TEST_TFIDF_FILENAME) and os.path.exists(TEST_TARGET_FILENAME)):
			pickle.dump(test_tfidf, open(TEST_TFIDF_FILENAME, 'wb'))
			pickle.dump(test_target, open(TEST_TARGET_FILENAME, 'wb'))


		# this is our classifier being trained!
		nb = MultinomialNB()
		clf = nb.fit(train_tfidf, train_target)

		# serialize and save classifier object to disk
		if not os.path.exists(PICKLED_CLASSIFIER_FILENAME):
			pickle.dump(clf, open(PICKLED_CLASSIFIER_FILENAME, 'wb'))

	else:
		try:
			clf = pickle.load(open(PICKLED_CLASSIFIER_FILENAME, 'rb'))
			test_tfidf = pickle.load(open(TEST_TFIDF_FILENAME, 'rb'))
			test_target = pickle.load(open(TEST_TARGET_FILENAME, 'rb'))
		except:
			print('\nPickled data not found on disk... try running program with -t option to train new model!')
			sys.exit(0)

	print('Running classifier against test set of 5000 reviews...')
	predicted = clf.predict(test_tfidf)

	error_dict = defaultdict(int)
	correct = 0
	for i in range(len(predicted)):
		if abs(predicted[i] - test_target[i]) <= 1:
			correct += 1
		else:
			error_dict[test_target[i]] += 1

	print('\nevaluation: {}% correct'.format(correct / len(predicted) * 100))
	print('\nAnalysis--number of mislabeled documents per label:')
	for k, v in sorted(error_dict.items()):
		if v:
			print('{} stars: {} misclassified'.format(k, v))
