import pandas as pd
from sklearn import svm

from sklearn.model_selection import KFold, train_test_split
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

df1 = pd.read_csv("weightxy.csv")
print(df1.shape)
print(df1.iloc[:,:-1].shape)
X_train, X_test, Y_train, Y_test = train_test_split(df1.iloc[:,:-1], df1['class_'], test_size = 0.3, random_state =42)

svm_clf = svm.SVC()

scores = cross_validate(svm_clf, X_train, Y_train, cv = 10)

print(scores.keys())

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
