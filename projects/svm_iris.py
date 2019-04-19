'''
Created on Apr 17, 2019

@author: dsj529
'''
from sklearn import datasets
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


#
#    Important parameters for SVC: gamma and C
#        gamma -> defines how far the influence of a single training example reaches
#                    Low value: influence reaches far      High value: influence reaches close
#
#        C -> trades off hyperplane surface simplicity + training examples missclassifications
#                    Low value: simple/smooth hyperplane surface 
#                    High value: all training examples classified correctly but complex surface 
dataset = datasets.load_iris()
features = dataset.data
target = dataset.target

feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.3)
model = svm.SVC().fit(feature_train, target_train)
preds = model.predict(feature_test)

print(confusion_matrix(target_test, preds))
print(accuracy_score(target_test, preds))
print(classification_report(target_test, preds))