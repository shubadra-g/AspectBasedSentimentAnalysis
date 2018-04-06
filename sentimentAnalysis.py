# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 16:09:25 2018

@author: Shubadra
"""

from nltk.corpus import stopwords
import string

def get_document(fname):
    f = open(fname, "r")
    data = f.read()
    print(data)
    f.close()
    return data

fname = 'data 1_train.csv'    
f = get_document(fname)
tokens = f.split(',')
# remove punctuations
table = str.maketrans('', '', string.punctuation) 
tokens = [w.translate(table) for w in tokens]
# remove tokens that are not alphbetic
tokens = [word for word in tokens if word.isalpha()]

# remove stop words
stop_words = set(stopwords.words('english'))
tokens = [w for w in tokens if not w in stop_words]
# filter out short tokens
tokens = [word for word in tokens if len(word) > 1]
print(tokens)
    