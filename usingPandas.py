import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
import string
import re
from sklearn.feature_extraction.text import CountVectorizer


# fname = 'project_2_train/' + 'data 2_train.csv'
# f = open(fname, 'r')
df = pd.read_csv('project_2_train/data 2_train.csv')
df = df[[' text', ' aspect_term', ' class']]
# print(df.head())

find = lambda searchList, elem: [[i for i, x in enumerate(searchList) if x == e] for e in elem]
# pfname = 'project_2_train/' + 'parsed_data.txt'
fname = 'project_2_train/' + 'data 2_train.csv'

stopwords_set = set(stopwords.words('english'))
stopwords_set = {'i', 'shan', 'just', 'how', 'each', 'out', 'themselves', 'their', 'before', 'were', 'very', 'as', 'further', 'his', 'a', 'once', 'youve', 'y', 'is', 'shouldve', 'youll', 'on', 'd', 'm', 'under', 'haven', 'which', 'only', 'them', 'was', 'by', 'needn', 'whom', 'that', 'when', 's', 'isn', 'its', 'no', 'wasn', 'in', 'we', 'theirs', 'those', 'this', 'having', 'and', 'ain', 'most', 'up', 'off', 'being', 'aren', 'shouldn', 'ourselves', 'from', 'down', 'herself', 'her', 'you', 'are', 'its', 'who', 'the', 'here', 'where', 'your', 'youd', 'she', 'didn', 'weren', 'about', 'has', 'our', 'an', 'yourselves', 'or', 'hasn', 'again', 'while', 'does', 'him', 'shes', 'above', 'below', 'itself', 'to', 'through', 'will', 'couldn', 'hers', 'they', 'doing', 'because', 'he', 'what', 'such', 'youre', 'nor', 'too', 'should', 'ours', 'then', 'himself', 'all', 'of', 'mightn', 'between', 'now', 'against', 'some', 'with', 'until', 'am', 'other', 'at', 'can', 'over', 'mustn', 'wouldn', 'do', 'for', 'after', 'hadn', 'me', 'been', 'same', 'doesn', 'my', 'these', 'll', 'did', 'had', 'it', 'so', 'ma', 'during', 'than', 'o', 'yourself', 'own', 'have', 're', 've', 'be', 'why', 't', 'there', 'more', 'won', 'yours', 'few', 'into', 'thatll', 'any', 'myself', 'both', 'don', 'if'}
# print(stopwords_set)
# stopwords_set.remove('but')
# stopwords_set.remove('not')
# pf = open(pfname, 'w')

X, Y = [], []
# print(X, Y)
def preprocess_data(line):
            # f = open(fname, 'r')
            # for i, line in enumerate(f):
            # if i != 0:
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
            columns[4] = columns[4].replace('\n', '')

f = open(fname, 'r')
#for i, line in enumerate(f):

count_vect = CountVectorizer(analyzer = preprocess_data).fit(df[' text'])
X_train_counts = count_vect.transform(count_vect)
X_train_counts.shape
