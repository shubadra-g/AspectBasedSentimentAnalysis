from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import numpy as np
import re
from stanfordcorenlp import StanfordCoreNLP
from functools import reduce

nlp = StanfordCoreNLP('http://localhost',port=9000)

# stopwords_set = set(stopwords.words('english'))
stopwords_set = {'i', 'shan', 'just', 'how', 'each', 'out', 'themselves', 'their', 'before', 'were', 'very', 'as', 'further', 'his', 'a', 'once', 'youve', 'y', 'is', 'shouldve', 'youll', 'on', 'd', 'm', 'under', 'haven', 'which', 'only', 'them', 'was', 'by', 'needn', 'whom', 'that', 'when', 's', 'isn', 'its', 'wasn', 'in', 'we', 'theirs', 'those', 'this', 'having', 'and', 'ain', 'most', 'up', 'off', 'being', 'aren', 'shouldn', 'ourselves', 'from', 'down', 'herself', 'her', 'you', 'are', 'its', 'who', 'the', 'here', 'where', 'your', 'youd', 'she', 'didn', 'weren', 'about', 'has', 'our', 'an', 'yourselves', 'or', 'hasn', 'again', 'while', 'does', 'him', 'shes', 'above', 'below', 'itself', 'to', 'through', 'will', 'couldn', 'hers', 'they', 'doing', 'because', 'he', 'what', 'such', 'youre', 'nor', 'too', 'should', 'ours', 'then', 'himself', 'all', 'of', 'mightn', 'between', 'now', 'against', 'some', 'with', 'until', 'am', 'other', 'at', 'can', 'over', 'mustn', 'wouldn', 'do', 'for', 'after', 'hadn', 'me', 'been', 'same', 'doesn', 'my', 'these', 'll', 'did', 'had', 'it', 'so', 'ma', 'during', 'than', 'o', 'yourself', 'own', 'have', 're', 've', 'be', 'why', 't', 'there', 'more', 'won', 'yours', 'few', 'into', 'thatll', 'any', 'myself', 'both', 'don', 'if'}

important_pos_tag_list_at = ['CC', 'JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'FW','IN', 'CD']
important_pos_tag_list_sent = ['CC', 'JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
sentence_weights = []
data_X, data_Y, data_X1 = [], [], []

file_name = 'project_2_train/' + 'data 2_train.csv'
file_obj = open(file_name, 'r')

for line_index, line in enumerate(file_obj):
    if line_index != 0:
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

        ''' POS Tagging the sentence'''
        pos_tagged_sentence = nlp.pos_tag(columns[1])
        pos_tagged_aspect_term = nlp.pos_tag(columns[2])

        ''' For aspect term: Removing the words whose POS tag is not important'''
        columns[2] = []
        for aspect_word_tag in pos_tagged_aspect_term:
            if aspect_word_tag[1] in important_pos_tag_list_at:
                columns[2].append(aspect_word_tag[0])

        '''If it removes every word in the aspect term then print'''
        if len(columns[2]) == 0:
            print(pos_tagged_aspect_term)

        ''' For sentence: Removing the words whose POS tag is not important'''
        columns[1] = []
        for word_tag in pos_tagged_sentence:
            if word_tag[1] in important_pos_tag_list_sent or word_tag[0] in columns[2]:
                columns[1].append(word_tag[0])

        '''If it removes every word in the sentence then print'''
        if len(columns[1]) == 0:
            print(pos_tagged_sentence)

        for word_index, word in enumerate(columns[1]):
            '''Remove special characters'''
            columns[1][word_index] = re.sub('[^0-9a-zA-Z]+', '', word).lower()
            ''' didn't is tokenized as did n't hence following code '''
            if columns[1][word_index] == 'nt':
                columns[1][word_index] = 'not'

        for word_index, word in enumerate(columns[2]):
            '''Remove special characters'''
            columns[2][word_index] = re.sub('[^0-9a-zA-Z]+', '', word).lower()
            ''' didn't is tokenized as did n't hence following code '''
            if columns[2][word_index] == 'nt':
                columns[2][word_index] = 'not'

        '''Remove empty string tokens'''
        columns[1] = [x.strip() for x in columns[1] if x.strip() != '']
        columns[2] = [x.strip() for x in columns[2] if x.strip() != '']

        ''' Remove stop words '''
        columns[2] = [word for word in columns[2] if word not in stopwords_set]
        columns[1] = [word for word in columns[1] if word not in stopwords_set]

        if len(columns[2]) == 0:
            print(pos_tagged_aspect_term)

        columns[1] = ' '.join(columns[1])
        columns[2] = ' '.join(columns[2])

        ''' The aspect term location given is not proper - hence extracting the location by ourselves '''
        columns[3] = []

        for m in re.finditer(columns[2], columns[1]):
            columns[3].append([m.start(), m.end()])

        ''' Tokenizing the words again '''
        columns[1] = columns[1].split(' ')
        columns[2] = columns[2].split(' ')

        ''' New column to specify the aspect term location in the list '''
        columns.append([])

        ''' for multiple positions of aspect term '''
        for aspect_term_loc in columns[3]:
            if aspect_term_loc[0] == 0: #start index of the 1st occurance of the aspect term
                columns[5].append([0])
            else:
                temp_len = aspect_term_loc[0] # assign the start position of the aspect term
                for word_index, word in enumerate(columns[1]):
                    temp_len -= (len(word) + 1) # Counting the words till the position - +1 for blank space
                    if temp_len == 0: # Reached the aspect word
                        columns[5].append([word_index+1])
                        break
            for k in range(len(columns[2])-1): # if multiple words in the aspect term - tag the following words in the sentence
                columns[5][-1].append(columns[5][-1][-1]+1)

        ''' if not found the aspect term location in the list then do '''
        if len(columns[5]) < 1:
#             print(columns)
#             print('---------------------------------')
            continue

        word_wt_list = [[word, 0] for word in columns[1]]

        ''' Assigning weights to every word based on the distance from the aspect term '''
        for j, elem in enumerate(columns[5]):
            for k, word in enumerate(columns[1]):
                if k < elem[0]: # Word left to the aspect term
                    dist = abs(elem[0] - k)
                    if word_wt_list[k][1] < 1/dist:
                        word_wt_list[k][1] = 1/dist
                elif k > elem[-1]: # word right to the aspect term : Aspect term can have multiple words
                    dist = abs(k - elem[-1])
                    if word_wt_list[k][1] < 1/dist:
                        word_wt_list[k][1] = 1/dist
            for aspect_word_loc in elem:
                word_wt_list[aspect_word_loc][1] = 1.5

        ''' For duplicate words - if it's aspect term - add the weights, if not, then take the weight which is greater, make the other weight as zero '''
        for j, word_1 in enumerate(word_wt_list):
            for k, word_2 in enumerate(word_wt_list[j+1:]):
                if word_1[0] == word_2[0]:
                    if word_1[1] == 1.5 and word_2[1] == 1.5:
                        # print(line)
                        # print(word_wt_list)
                        word_1[1] += 1.5
                        word_2[1] = 0
                    elif word_1[1] < word_2[1]:
                        # print(line)
                        # print(word_wt_list)
                        word_1[1] = word_2[1]
                        word_2[1] = 0
                    elif word_1[1] > word_2[1]:
                        # print(line)
                        # print(word_wt_list)
                        word_2[1] = 0
                    elif word_1[1] == word_2[1]:
                        word_2[1] = 0


        ''' For removing duplicate words - they have weight zero now'''
        word_wt_list = [elem for elem in word_wt_list if elem[1] != 0]

        if len(columns[2]) > 1 and len(columns[5]) > 1:
#             print(line)
#             print(columns)
#             print(word_wt_list)
            pass

        if columns[0] == '2911_0':
#             print(line)
#             print(word_wt_list)
            pass

        '''Removing new line character from columns[4] -  the class column'''
        columns[4] = columns[4].rstrip('\n')

        data_X.append(columns[1])
        data_X1.append(' '.join(columns[1]))

        data_Y.append(columns[4])
        sentence_weights.append(word_wt_list)

    if line_index == 100:
#         break
        pass

file_obj.close()
nlp.close()
'''Building vocabulary and counting word occurances'''

X_reduced = reduce(lambda x1, x2: x1 + x2, data_X)
vocab = list(set(X_reduced))
print(len(vocab))

weight_v = np.zeros_like(vocab, dtype = np.float_)
weight_x = []
pf = open('weightxy_data2.csv', 'w')

''' Writing column labels for pandas dataframe'''
pf.write(','.join(vocab))
pf.write(',class_')
pf.write('\n')

for i, sentence in enumerate(sentence_weights):
    for word in sentence:
        v_index = vocab.index(word[0])
        weight_v[v_index] = word[1]

    weight_x.append(weight_v)
    if len(sentence) != weight_v[np.where(weight_v > 0)].shape[0]:
        print(sentence)
        print(len(sentence))
        print(weight_v[np.where(weight_v > 0)])

    ''' code to write the data to file '''
    temp = [str(j) for j in weight_v.tolist()]
    pf.write(','.join(temp))
    pf.write(','+str(data_Y[i]))
    pf.write('\n')
    weight_v = np.zeros_like(weight_v)

pf.close()

weight_x = np.array(weight_x)

print(weight_x.shape)
