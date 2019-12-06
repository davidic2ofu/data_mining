# import ijson
from collections import defaultdict
import json
import numpy as np
import os
from time import time


dataset_path = os.path.join(
	os.getcwd(), # os.path.dirname(__file__),
	'dataset',
)

datasets = {
	f.split('.')[0]: dataset_path + os.sep + f for f in filter(lambda x: x.endswith('.json'), os.listdir(dataset_path))
}

before = time()
reviews = defaultdict(list)
with open(datasets['review']) as f:
	for line in f:
		contents = json.loads(line)
		reviews[contents['stars']].append(contents['text'])

print(time() - before)

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


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
train_tfidf = vectorizer.fit_transform(balanced_reviews)
train_target = balanced_ratings


# from sklearn.svm import SVC
# svm = SVC()

# before = time()
# clf = svm.fit(train_tfidf, train_target)
# print(time() - before)

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

before = time()
clf = nb.fit(train_tfidf, train_target)
print(time() - before)


test_tfidf = vectorizer.transform(test_reviews)
test_target = test_ratings

predicted = clf.predict(test_tfidf)

correct = 0
for i in range(len(predicted)):
	if abs(predicted[i] - test_target[i]) <= 1:
		correct += 1

print('evaluation: {}% correct'.format(correct / len(predicted) * 100))






import itertools

all_cats = []
for catlist in itertools.chain(*list(businesses.values())):
	if catlist and 'Restaurants' in catlist and 'IT Services & Computer Repair' in catlist:
		catlist

		for cat in catlist.split(', '):
			all_cats.append(cat)

unique_categories = list(set(all_cats))




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


import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(400)

import nltk
nltk.download('wordnet')


'''
Step 2: Data Preprocessing
We will perform the following steps:

Tokenization: Split the text into sentences and the sentences into words. Lowercase the words and remove punctuation.
Words that have fewer than 3 characters are removed.
All stopwords are removed.
Words are lemmatized - words in third person are changed to first person and verbs in past and future tenses are changed into present.
Words are stemmed - words are reduced to their root form.
'''



# some extra stop words (not used)
food_stopwords = ['chicken', 'curry', 'steak', 'egg', 'pork', 'meat',
                 'sandwich', 'cheese', 'pasta', 'salad', 'taco', 'salsa',
                 'guacamole', 'bruschetta', 'fish', 'dessert', 'onion',
                 'bun', 'sushi', 'sashimi', 'shrimp', 'crab', 'seafood',
                 'lobster', 'meatball', 'potato', 'entree', 'burrito',
                 'tortilla', 'food', 'olive', 'ramen', 'rib', 'brisket',
                 'bbq', 'bean', 'chip', 'mac', 'rice', 'beef', 'avocado', 
                 'pizza', 'garlic', 'crust', 'burger', 'bacon', 'meal',
                 'toast', 'bread', 'lunch', 'breakfast', 'appetizer',
                 'filet', 'cake', 'sauce', 'dish', 'dining', 'pie',
                 'nacho', 'enchilada', 'wing', 'roll', 'salmon', 'oyster',
                 'soup', 'sausage', 'truffle', 'noodle', 'ravioli', 'lasagna',
                 'veal', 'buffet', 'tiramisu', 'eggplant', 'chocolate', 'scallop',
                 'chef', 'duck', 'butter', 'steakhouse', 'kobe', 'caviar',
                 'stroganoff', 'corn', 'mushroom', 'thai', 'prawn', 'coconut',
                 'pretzel', 'pho', 'tuna', 'donut', 'chili', 'panini', 'fig',
                 'holstein', 'calamari', 'pancake', 'fruit', 'pierogi', 'pierogis',
                 'pierogies', 'mignon', 'rare', 'medium', 'lamb', 'milkshake',
                 'ribeye', 'mashed', 'bone', 'bass', 'sea', 'guac', 'queso',
                 'fajitas', 'carne', 'pasty', 'asada', 'mozzarella', 'marsala',
                 'spaghetti', 'gnocchi', 'parm', 'alfredo', 'linguine', 'buffalo',
                 'falafel', 'hummus', 'pita', 'scrambled', 'risotto', 'fat',
                 'strip', 'roast', 'miso', 'tempura', 'udon', 'edamame',
                 'cucumber', 'dipping', 'yellowtail', 'waffle', 'quesadilla', 'dog',
                 'primanti', 'tot', 'tater', 'phyllo', 'pomegranate', 'cinnamon',
                 'shepherd', 'banger', 'corned', 'foie', 'gras', 'latte', 'banana',
                 'poutine', 'seabass', 'du', 'je', 'au', 'mais', 'très', 'asparagus',
                 'slider', 'tikka', 'naan', 'popcorn', 'masala', 'bonefish', 'lime']

additional_stopwords = ['restaurant', 'vegas', 'waitress', 'dinner',
                        'wa', 'waiter', 'scottsdale', 'toronto',
                       'pittsburgh', 'madison', 'fremont', 'manager',
                       'husband', 'phoenix', 'dakota', 'caesar', 
                       'bellagio', 'canal', 'venetian', 'mandalay',
                       'lotus', 'siam', 'buca','beppo',
                       'di', 'buca', 'ohio', 'tretmont',
                       'bathroom', 'montreal', 'italy', 'et', 'est',
                       'que', 'il', 'en', 'la', 'une', 'pa', 'hostess']



# Write a function to perform the pre processing steps on the entire dataset
def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


more_stopwords = [lemmatize_stemming(word) for word in food_stopwords + additional_stopwords]
all_stopwords = STOPWORDS + more_stopwords


# Tokenize and lemmatize
def preprocess(text):
    result=[]
    for token in simple_preprocess(text, deacc=True, min_len=4):
        if token not in STOPWORDS:
            result.append(lemmatize_stemming(token))
    return result



'''
Preview a document after preprocessing
'''
doc_sample = all_the_reviews[0]

print("Original document: ")
words = [word for word in doc_sample.split(' ')]
print("\nBefore preprocessing: ")
print(words)
print("\nTokenized and lemmatized document: ")
print(preprocess(doc_sample))



'''
Need to split reviews into training and testing
'''

training_reviews = all_the_reviews[:50000]

processed_reviews = [preprocess(review) for review in training_reviews]


'''
Step 3: Bag of words on the dataset¶
Create a dictionary from 'processed_docs' containing the number of times a word appears
in the training set. To do that, we pass processed_docs to gensim.corpora.Dictionary()
'''

dictionary = gensim.corpora.Dictionary(processed_reviews)

'''
Filter out tokens that appear either less than 15 times or more than 10% of documents
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

print('Original document:')
print(training_reviews[25])

for doc in bow_corpus[25]:
    print("Word {} (\"{}\") appears {} time.".format(doc[0], dictionary[doc[0]], doc[1]))



'''
Step 4: Running LDA using Bag of Words
We are going for 10 topics in the document corpus.

We will be running LDA using all CPU cores to parallelize and speed up model training.
'''

def topics(n):
	lda_model =  gensim.models.LdaMulticore(
		bow_corpus,
		num_topics = n,
		id2word = dictionary,
		passes = 5,
		workers = 8,
	)
	for i, topic in lda_model.print_topics(
		num_topics=-1,
		num_words=20,
	):
	    print("Topic: {} \nWords: {}".format(i, topic))



'''
TOPICS WITH n=14

Topic: 0 	Bars
Words: 0.029*"pizza" + 0.014*"beer" + 0.013*"wine" + 0.011*"wing" + 0.009*"night" + 0.008*"select" + 0.007*"visit" + 0.006*"coffe" + 0.006*"busi" + 0.006*"atmospher"
Topic: 1 	American
Words: 0.028*"sandwich" + 0.025*"fri" + 0.021*"burger" + 0.010*"chees" + 0.008*"grill" + 0.008*"breakfast" + 0.007*"clean" + 0.007*"serv" + 0.006*"awesom" + 0.006*"potato"
Topic: 2 	Coffee shop
Words: 0.073*"cake" + 0.018*"cooki" + 0.013*"bakeri" + 0.012*"birthday" + 0.012*"chocol" + 0.012*"baker" + 0.010*"wed" + 0.009*"coffe" + 0.007*"free" + 0.007*"breakfast"
Topic: 3 	Short order
Words: 0.016*"say" + 0.015*"ask" + 0.015*"minut" + 0.015*"tell" + 0.014*"custom" + 0.013*"manag" + 0.012*"take" + 0.009*"leav" + 0.008*"busi" + 0.008*"waitress"
Topic: 4 	Fancy
Words: 0.008*"night" + 0.008*"hour" + 0.007*"say" + 0.007*"seat" + 0.006*"take" + 0.006*"soup" + 0.006*"leav" + 0.005*"give" + 0.005*"star" + 0.005*"lobster"
Topic: 5 	Casino Diner
Words: 0.054*"buffet" + 0.016*"crab" + 0.016*"vega" + 0.014*"bellagio" + 0.012*"line" + 0.011*"select" + 0.011*"dinner" + 0.011*"dessert" + 0.011*"qualiti" + 0.010*"leg"
Topic: 6 	Indian
Words: 0.014*"meal" + 0.014*"excel" + 0.011*"perfect" + 0.011*"flavor" + 0.009*"indian" + 0.009*"appet" + 0.009*"wine" + 0.007*"enjoy" + 0.007*"salad" + 0.007*"dessert"
Topic: 7 	Korean/Hibachi
Words: 0.019*"server" + 0.011*"dinner" + 0.011*"cook" + 0.009*"take" + 0.007*"enjoy" + 0.007*"serv" + 0.007*"korean" + 0.006*"owner" + 0.006*"hibachi" + 0.006*"work"
Topic: 8 	Sushi
Words: 0.022*"soup" + 0.019*"sushi" + 0.013*"wrap" + 0.012*"ramen" + 0.010*"bowl" + 0.010*"roll" + 0.009*"quick" + 0.008*"area" + 0.008*"locat" + 0.007*"vietnames"
Topic: 9 	Italian
Words: 0.031*"salad" + 0.020*"pizza" + 0.018*"bread" + 0.014*"chees" + 0.013*"pasta" + 0.011*"meat" + 0.011*"portion" + 0.008*"gyro" + 0.007*"garlic" + 0.007*"tomato"
Topic: 10 	French
Words: 0.058*"crepe" + 0.028*"cream" + 0.026*"sweet" + 0.025*"dessert" + 0.012*"waffl" + 0.012*"latt" + 0.010*"green" + 0.010*"chocol" + 0.009*"potato" + 0.008*"flavour"
Topic: 11 	Mexican
Words: 0.030*"taco" + 0.009*"night" + 0.009*"review" + 0.008*"room" + 0.007*"pizza" + 0.007*"margarita" + 0.007*"salsa" + 0.007*"asada" + 0.006*"mexican" + 0.006*"awesom"
Topic: 12 	Thai
Words: 0.069*"thai" + 0.014*"curri" + 0.014*"hous" + 0.011*"roll" + 0.009*"spici" + 0.008*"soup" + 0.008*"charlott" + 0.008*"rice" + 0.007*"area" + 0.007*"spring"
Topic: 13 	Chinese
Words: 0.020*"rice" + 0.017*"noodl" + 0.015*"fri" + 0.012*"pork" + 0.012*"roll" + 0.011*"flavor" + 0.011*"spici" + 0.010*"beef" + 0.009*"fish" + 0.009*"soup"
'''



import pickle

pickle.dump(lda_model, open('lda_model.p', 'wb'))

lda_model2 = pickle.load(open('lda_model.p', 'rb'))




from gensim.test.utils import datapath

temp_file = datapath("model")
lda_model.save(temp_file)

lda_model3 = gensim.models.LdaMulticore.load(temp_file)

