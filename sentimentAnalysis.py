# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 16:09:25 2018

@author: Shubadra
"""

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import string
import re

find = lambda searchList, elem: [[i for i, x in enumerate(searchList) if x == e] for e in elem]
fname = 'project_2_train/' + 'dara 1_train.csv'

stopwords_set = set(stopwords.words('english'))
stopwords_set = {'i', 'shan', 'just', 'how', 'each', 'out', 'themselves', 'their', 'before', 'were', 'very', 'as', 'further', 'his', 'a', 'once', 'youve', 'y', 'is', 'shouldve', 'youll', 'on', 'd', 'm', 'under', 'haven', 'which', 'only', 'them', 'was', 'by', 'needn', 'whom', 'that', 'when', 's', 'isn', 'its', 'no', 'wasn', 'in', 'we', 'theirs', 'those', 'this', 'having', 'and', 'ain', 'most', 'up', 'off', 'being', 'aren', 'shouldn', 'ourselves', 'from', 'down', 'herself', 'her', 'you', 'are', 'its', 'who', 'the', 'here', 'where', 'your', 'youd', 'she', 'didn', 'weren', 'about', 'has', 'our', 'an', 'yourselves', 'or', 'hasn', 'again', 'while', 'does', 'him', 'shes', 'above', 'below', 'itself', 'to', 'through', 'will', 'couldn', 'hers', 'they', 'doing', 'because', 'he', 'what', 'such', 'youre', 'nor', 'too', 'should', 'ours', 'then', 'himself', 'all', 'of', 'mightn', 'between', 'now', 'against', 'some', 'with', 'until', 'am', 'other', 'at', 'can', 'over', 'mustn', 'wouldn', 'do', 'for', 'after', 'hadn', 'me', 'been', 'same', 'doesn', 'my', 'these', 'll', 'did', 'had', 'it', 'so', 'ma', 'during', 'than', 'o', 'yourself', 'own', 'have', 're', 've', 'be', 'why', 't', 'there', 'more', 'won', 'yours', 'few', 'into', 'thatll', 'any', 'myself', 'both', 'don', 'if'}
# print(stopwords_set)
# stopwords_set.remove('but')
# stopwords_set.remove('not')

f = open(fname, 'r')
for i, line in enumerate(f):
    if i != 0:
        columns = line.split(',')
        columns[1] = columns[1].replace('[comma]', '')
        columns[2] = columns[2].replace('[comma]', '')
        tokenizer = RegexpTokenizer(r'\w+') # doesn't work if the special char is in the token
        columns[1] = tokenizer.tokenize(columns[1].lower())
        columns[2] = tokenizer.tokenize(columns[2].lower())
        columns[3] = []
        columns[1] = [word for word in columns[1] if word not in stopwords_set]
        columns[2] = [word for word in columns[2] if word not in stopwords_set]

        columns[1] = ' '.join(columns[1])
        columns[2] = ' '.join(columns[2])

        # columns[3] = columns[1].count(columns[2])
        for m in re.finditer(columns[2], columns[1]):
            columns[3].append([m.start(), m.end()])
        # print(columns)
        columns[1] = columns[1].split(' ')
        columns[2] = columns[2].split(' ')
        columns.append([])
        # columns[5] = []
        for elem in columns[3]:
            if elem[0] == 0: #start index of the 1st occurance of the aspect term
                columns[5].append([0])
            else:
                temp_len = elem[0]
                for j, tokens in enumerate(columns[1]):
                    temp_len -= (len(tokens) + 1) # +1 for blank space
                    if temp_len == 0:
                        columns[5].append([j+1])
                        break
            # for j, elem2 in enumerate(columns[2]): #aspect term column
            #     if j > 0:
            #         columns[5][-1].append(columns[5][-1][-1]+1)
            for k in range(len(columns[2])-1):
                columns[5][-1].append(columns[5][-1][-1]+1)
        if len(columns[3]) > 1:
            # print(columns[1],"      ", columns[2], "    ", columns[3])
            print(line)
            print(columns)
            # pass

        # columns[3] = find(columns[1], columns[2])
        # for elem in columns[3]:
        #     if len(elem) > 1:
        #         # print(columns[3])
        #         pass

        # print(columns[1])
        # print(columns[1].index(columns[2][0]))
        # print(columns[1])
        # word_tokens = word_tokenize(columns[1])

        # print(columns)
        if i == 4:
            # pass
            break

# tokens = f.split(',') # columns
# remove punctuations
# table = str.maketrans('', '', string.punctuation)
# print(table)
# tokens = [w.translate(table) for w in tokens]
# # remove tokens that are not alphbetic
# tokens = [word for word in tokens if word.isalpha()]
#
# # remove stop words
# stop_words = set(stopwords.words('english'))
# tokens = [w for w in tokens if not w in stop_words]
# # filter out short tokens
# tokens = [word for word in tokens if len(word) > 1]
# print(tokens)
