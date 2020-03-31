# Doc2Vec Text Classification [![Build Status](https://travis-ci.org/ibrahimsharaf/doc2vec.svg?branch=master)](https://travis-ci.org/ibrahimsharaf/doc2vec)

Text classification model which uses gensim Doc2Vec for generating paragraph embeddings and scikit-learn Logistic Regression for classification.


### Dataset

25,000 IMDB movie reviews, specially selected for sentiment analysis. The sentiment of reviews is binary (1 for postive, 0 for negative).

This source dataset was collected in association with the following publication:

```Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011). "Learning Word Vectors for Sentiment Analysis." The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).```

### Usage
- Install the required tools 

    ```pip install -r requirements.txt```
- Run the script 
    
     ```python text_classifier.py```

### References
- Kaggle – Bag of Words Meets Bags of Popcorn (https://www.kaggle.com/c/word2vec-nlp-tutorial)
- Gensim – Deep learning with paragraph2vec (https://radimrehurek.com/gensim/models/doc2vec.html)
- Quoc Le and Tomas Mikolov. Distributed Representations of Sentences and Documents (https://arxiv.org/pdf/1405.4053v2.pdf)
