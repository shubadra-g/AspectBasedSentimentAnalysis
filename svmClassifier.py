from sklearn import svm
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
# list of text documents
fname = 'project_2_train/' + 'data 2_train.csv'
text = open(fname, 'r')
df1 = pd.read_csv('project_2_train/data 2_train.csv')
df1 = df[[' text', ' aspect_term', ' class']]
df2 = pd.read_csv('project_2_train/dara 1_train.csv')
df2 = df[[' text', ' aspect_term', ' class']]

stopwords.words('english')
X = df1.drop(' class', axis = 1)
Y = df1[' class']

X_train, X_test, label_train, label_test = train_test_split(df1[' text'], Y, test_size = 0.3, random_state =42)
