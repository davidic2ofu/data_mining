## Document Classification and Topic Modeling Using the Yelp Dataset

### Data Mining Project Fall 2019

```https://github.com/davidic2ofu/data_mining```


These programs expect Python3 environment (you may want to use a virtualenv).

To install package requirements please run

```pip install -r requirements.txt```

### LDA Topic Modeling

To use my pretrained model for LDA topic modeling:

```python topic_modeling.py```

Please also train your own if you have the Yelp dataset saved in the ```dataset``` directory:

```python topic_modeling.py -t```

add ```-n``` to specify number of topics (default is 14 -- I had good luck with 14!)

```python topic_modeling.py -t -n 25```

### Naive Bayes Document Classification

To use my pretrained model for document classification and analysis:

```python document_classification.py```

Please also train your own if you have the Yelp dataset saved in the ```dataset``` directory:

```python document_classification.py -t```
