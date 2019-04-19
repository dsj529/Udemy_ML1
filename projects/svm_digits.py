'''
Created on Apr 17, 2019

@author: dsj529
'''
from sklearn import datasets, svm, metrics
from sklearn.metrics import accuracy_score, classification_report

import matplotlib.pyplot as plt


# Import datasets, classifiers and performance metrics
digits = datasets.load_digits()
images_and_labels = list(zip(digits.images, digits.target))

# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

clf = svm.SVC(gamma=0.001)
trainTestSplit = int(n_samples*0.75)
clf.fit(data[:trainTestSplit], digits.target[:trainTestSplit])

expected = digits.target[trainTestSplit:]
predicted = clf.predict(data[trainTestSplit:])

print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
print(accuracy_score(expected, predicted))


# let's test on the last few images
plt.imshow(digits.images[-2], cmap=plt.cm.gray_r, interpolation='nearest')
print("Prediction for test image: ", clf.predict(data[-2].reshape(1,-1)))

plt.show()
