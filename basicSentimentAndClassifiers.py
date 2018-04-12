# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 16:09:25 2018

@author: Shubadra
"""

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
import string
import re
from sklearn.feature_extraction.text import CountVectorizer
from functools import reduce
from sklearn.model_selection import KFold, train_test_split
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
# from sklearn.metrics import accuracy_score
import pickle
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_validate
from sklearn.metrics.scorer import make_scorer
from sklearn.metrics import confusion_matrix
from sklearn import metrics



find = lambda searchList, elem: [[i for i, x in enumerate(searchList) if x == e] for e in elem]
# pfname = 'project_2_train/' + 'parsed_data.txt'
fname = 'project_2_train/' + 'data 2_train.csv'

stopwords_set = set(stopwords.words('english'))
stopwords_set = {'i', 'shan', 'just', 'how', 'each', 'out', 'themselves', 'their', 'before', 'were', 'very', 'as', 'further', 'his', 'a', 'once', 'youve', 'y', 'is', 'shouldve', 'youll', 'on', 'd', 'm', 'under', 'haven', 'which', 'only', 'them', 'was', 'by', 'needn', 'whom', 'that', 'when', 's', 'isn', 'its', 'no', 'wasn', 'in', 'we', 'theirs', 'those', 'this', 'having', 'and', 'ain', 'most', 'up', 'off', 'being', 'aren', 'shouldn', 'ourselves', 'from', 'down', 'herself', 'her', 'you', 'are', 'its', 'who', 'the', 'here', 'where', 'your', 'youd', 'she', 'didn', 'weren', 'about', 'has', 'our', 'an', 'yourselves', 'or', 'hasn', 'again', 'while', 'does', 'him', 'shes', 'above', 'below', 'itself', 'to', 'through', 'will', 'couldn', 'hers', 'they', 'doing', 'because', 'he', 'what', 'such', 'youre', 'nor', 'too', 'should', 'ours', 'then', 'himself', 'all', 'of', 'mightn', 'between', 'now', 'against', 'some', 'with', 'until', 'am', 'other', 'at', 'can', 'over', 'mustn', 'wouldn', 'do', 'for', 'after', 'hadn', 'me', 'been', 'same', 'doesn', 'my', 'these', 'll', 'did', 'had', 'it', 'so', 'ma', 'during', 'than', 'o', 'yourself', 'own', 'have', 're', 've', 'be', 'why', 't', 'there', 'more', 'won', 'yours', 'few', 'into', 'thatll', 'any', 'myself', 'both', 'don', 'if'}
# print(stopwords_set)
# stopwords_set.remove('but')
# stopwords_set.remove('not')
# pf = open(pfname, 'w')

X, Y, X1 = [], [], []
# print(X, Y)
f = open(fname, 'r')
for i, line in enumerate(f):
    if i != 0:
        # print('Processing data...')
        ''' Splitting the columns based on comma - since it is csv'''
        columns = line.split(',')

        ''' The comma in the actual sentence was represented as [commma] because of csv format, replace that '''
        columns[1] = columns[1].replace('[comma]', ',')
        columns[2] = columns[2].replace('[comma]', ',')

        ''' Some aspect terms are mic in the sentence where computer/mic is present '''
        columns[1] = columns[1].replace('-', ' ')
        columns[2] = columns[2].replace('-', ' ')
        columns[1] = columns[1].replace('/', ' ')
        columns[2] = columns[2].replace('/', ' ')

        ''' NOt used anymore'''
        # tokenizer = RegexpTokenizer(r'\w+') # doesn't work if the special char is in the token
        # columns[2] = tokenizer.tokenize(columns[2].lower())

        '''Tokenize the words'''
        columns[1] = word_tokenize(columns[1])
        columns[2] = word_tokenize(columns[2])

        '''For sentence'''
        for j, elem in enumerate(columns[1]):
            '''Remove special characters'''
            columns[1][j] = re.sub('[^0-9a-zA-Z]+', '', elem).lower()
            ''' didn't is tokenized as did n't hence following code '''
            if columns[1][j] == 'nt':
                columns[1][j] = 'not'

        '''For aspect term'''
        for j, elem in enumerate(columns[2]):
            '''Remove special characters'''
            columns[2][j] = re.sub('[^0-9a-zA-Z]+', '', elem).lower()
            ''' didn't is tokenized as did n't hence following code '''
            if columns[2][j] == 'nt':
                columns[2][j] = 'not'

        '''Remove empty string tokens'''
        columns[1] = [x for x in columns[1] if x != '']
        columns[2] = [x for x in columns[2] if x != '']

        ''' Remove stop words '''
        columns[1] = [word for word in columns[1] if word not in stopwords_set]
        columns[1] = [word for word in columns[1] if word not in stopwords_set]
        columns[2] = [word for word in columns[2] if word not in stopwords_set]

        columns[1] = ' '.join(columns[1])
        columns[2] = ' '.join(columns[2])

        ''' The aspect term location given is not proper - hence extracting the location by ourselves '''
        columns[3] = []
        ''' Finds the location of the aspect term in the sentence '''
        for m in re.finditer(columns[2], columns[1]):
            columns[3].append([m.start(), m.end()])
            ''' since some aspect terms are not surrounded by spaces, hence below code '''
            if m.start() != 0 and columns[1][m.start() - 1] != ' ':
                columns[1] = columns[1][:m.start()] + ' ' + columns[1][m.start():]

        ''' Because of adding space in above code, the position is messed up, hence redoing it. '''
        columns[3] = []

        for m in re.finditer(columns[2], columns[1]):
            columns[3].append([m.start(), m.end()])

        ''' Tokenizing the words again '''
        columns[1] = columns[1].split(' ')
        columns[2] = columns[2].split(' ')
        ''' New column to specify the aspect term location in the list '''
        columns.append([])
        # columns[5] = []
        ''' for multiple positions of the columns '''
        for elem in columns[3]:
            if elem[0] == 0: #start index of the 1st occurance of the aspect term
                columns[5].append([0])
            else:
                temp_len = elem[0] # assign the start position of the aspect term
                for j, tokens in enumerate(columns[1]):
                    temp_len -= (len(tokens) + 1) # Counting the words till the position - +1 for blank space
                    if temp_len == 0: # Reached the aspect word
                        columns[5].append([j+1])
                        break
            for k in range(len(columns[2])-1): # if multiple words in the aspect term - tag the following words in the sentence
                columns[5][-1].append(columns[5][-1][-1]+1)

        ''' if not found the aspect term location in the list then do'''
        if len(columns[5]) < 1:
            print(line)
            print(columns)
            pass

        # if i == 10:
        #     # break
        #     pass
        # # print('***********')
        '''Removing new line character from columns[4] -  the polarity column'''
        columns[4] = columns[4].rstrip('\n')
        X.append(columns[1])
        X1.append(' '.join(columns[1]))
        Y.append(columns[4])

        # print("Values in X: \n\n", X)
        # print("Values in Y: \n\n", Y)

'''Building vocabulary and counting word occurances'''
X_reduced = reduce(lambda x1, x2: x1 + x2, X)
X_ = list(set(X_reduced))

count_vect = CountVectorizer(vocabulary = X_)
X_train_counts = count_vect.fit_transform(X1).toarray()

X_train, X_test, Y_train, Y_test = train_test_split(X_train_counts, Y, test_size = 0.3, random_state = 42)
# print(len(X_train), len(label_train))
'''K-Fold Cross Validation'''
# kf = KFold(n_splits = 10)
# print(kf.get_n_splits(X))
# print(kf)
X = np.asarray(X_train_counts)
# print(X[0])
# X_train, X_test = np.asarray(X_train), np.asarray(X_test)
# Y_train, Y_test = np.asarray(Y_train), np.asarray(Y_test)
Y = np.asarray(Y)

''' Using SVM to classify'''
svm_clf = svm.SVC()
scores = cross_validate(svm_clf, X_train, Y_train, cv = 10)

# Saving the model
mod_filename = 'svm_model.sav'
pickle.dump(svm_clf, open(mod_filename, 'wb'))

svm_clf.fit(X_train, Y_train)
t_pred = svm_clf.predict(X_test)
mod_accracy = accuracy_score(Y_test, t_pred)

t_precision = precision_score(Y_test, t_pred, average='weighted')
t_recall = recall_score(Y_test, t_pred, average='weighted')
t_f1_score = f1_score(Y_test, t_pred, average='weighted')

print('*** Scores for SVM: ***')
target_names = ['0', '1', '-1']
print('Pr, Re scores wrt each class')
print(classification_report(Y_test, t_pred, target_names=target_names))

print('\n\nModel Accuracy is: {:.4%}' .format(mod_accracy))
print('\n\nPrecision Score is: {:.4%}' .format(t_precision))
print('\n\nRecall Score is: {:.4%}' .format(t_recall))
print('\n\nF1-Score is: {:.4%}' .format(t_f1_score))

# Using MNB:
from sklearn.naive_bayes import MultinomialNB
mnb_clf = MultinomialNB()
scores = cross_validate(mnb_clf, X_train, Y_train, cv = 10)

mnb_clf.fit(X_train, Y_train)
t_pred = mnb_clf.predict(X_test)
mod_accracy = accuracy_score(Y_test, t_pred)

# Saving the model
mod_filename = 'mnb_model.sav'
pickle.dump(mnb_clf, open(mod_filename, 'wb'))

t_precision = precision_score(Y_test, t_pred, average='weighted')
t_recall = recall_score(Y_test, t_pred, average='weighted')
t_f1_score = f1_score(Y_test, t_pred, average='weighted')

print('*** Scores for MNB: ***')

target_names = ['0', '1', '-1']
print('Pr, Re scores wrt each class')
print(classification_report(Y_test, t_pred, target_names=target_names))

print('\n\nModel Accuracy is: {:.4%}' .format(mod_accracy))
print('\n\nPrecision Score is: {:.4%}' .format(t_precision))
print('\n\nRecall Score is: {:.4%}' .format(t_recall))
print('\n\nF1-Score is: {:.4%}' .format(t_f1_score))

# save_model = pickle.dumps(clf_mnb)


# Using RandomForest:

from sklearn.ensemble import RandomForestClassifier
randomForest_clf = RandomForestClassifier(max_depth=2, random_state=4)
randomForest_clf.fit(X_train, Y_train)
scores = cross_validate(randomForest_clf, X_train, Y_train, cv = 10)

randomForest_clf.fit(X_train, Y_train)
t_pred = randomForest_clf.predict(X_test)
mod_accracy = accuracy_score(Y_test, t_pred)

# Saving the model
mod_filename = 'randomForest_model.sav'
pickle.dump(randomForest_clf, open(mod_filename, 'wb'))

t_precision = precision_score(Y_test, t_pred, average='weighted')
t_recall = recall_score(Y_test, t_pred, average='weighted')
t_f1_score = f1_score(Y_test, t_pred, average='weighted')

print('*** Scores for Random Forest: ***')

target_names = ['0', '1', '-1']
print('Pr, Re scores wrt each class')
print(classification_report(Y_test, t_pred, target_names=target_names))

print('\n\nModel Accuracy is: {:.4%}' .format(mod_accracy))
print('\n\nPrecision Score is: {:.4%}' .format(t_precision))
print('\n\nRecall Score is: {:.4%}' .format(t_recall))
print('\n\nF1-Score is: {:.4%}' .format(t_f1_score))


# AdaBoost:

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

adaBoost_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2)) #, n_estimators=2, learning_rate=1)
adaBoost_clf.fit(X_train, Y_train)
scores = cross_validate(adaBoost_clf, X_train, Y_train, cv = 10)

# adaBoost_clf.fit(X_train, Y_train)
t_pred = adaBoost_clf.predict(X_test)
mod_accracy = accuracy_score(Y_test, t_pred)

# Saving the model
mod_filename = 'adaBoost_model.sav'
pickle.dump(adaBoost_clf, open(mod_filename, 'wb'))

t_precision = precision_score(Y_test, t_pred, average='weighted')
t_recall = recall_score(Y_test, t_pred, average='weighted')
t_f1_score = f1_score(Y_test, t_pred, average='weighted')

print('*** Scores for Ada Boost: ***')

target_names = ['0', '1', '-1']
print('Pr, Re scores wrt each class')
print(classification_report(Y_test, t_pred, target_names=target_names))

print('\n\nModel Accuracy is: {:.4%}' .format(mod_accracy))
print('\n\nPrecision Score is: {:.4%}' .format(t_precision))
print('\n\nRecall Score is: {:.4%}' .format(t_recall))
print('\n\nF1-Score is: {:.4%}' .format(t_f1_score))
