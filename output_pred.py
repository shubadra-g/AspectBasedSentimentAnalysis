
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
import numpy as np
import string
import re
from functools import reduce

fname = 'project_2_train/' + 'Data-2_test.csv'
f = open(fname, 'r')

output_file = 'Mohit_Adwani_Shubadra_Govindan_Data-2.txt'
o_fname = open(output_file, 'w')
line_count = 0
for ind in f:
    line_count += 1
# print(line_count)
f.close()

f = open(fname, 'r')
for i, line in enumerate(f):
    if i != 0:
      ''' Splitting the columns based on comma - since it is csv'''
      columns = line.split(',')
      # print(columns[0])
      output_label = 1
      # print(i)
      o_fname.write('%s' %(columns[0]) + ';;' + '%d' %(output_label))
      if i != line_count - 1:
        o_fname.write('\n')
f.close()
o_fname.close()
