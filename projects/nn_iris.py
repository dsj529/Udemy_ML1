'''
Created on Apr 25, 2019

@author: dsj529
'''
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

dataset = datasets.load_iris()

features = dataset.data
y = dataset.target.reshape(-1,1)

encoder = OneHotEncoder()
targets = encoder.fit_transform(y)

train_features, test_features, train_targets, test_targets = train_test_split(features,targets, test_size=0.2)

model = Sequential()
# first parameter is output dimension
model.add(Dense(10, input_dim=4, activation='relu'))
model.add(Dense(10, input_dim=10, activation='relu'))
model.add(Dense(10, input_dim=10, activation='relu'))
model.add(Dense(3, activation='softmax'))

optimizer = Adam(lr=0.005)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

model.fit(train_features, train_targets, epochs=10000, batch_size=20, verbose=2)

results = model.evaluate(test_features, test_targets)

print("Accuracy on the test dataset: %.2f" % results[1])