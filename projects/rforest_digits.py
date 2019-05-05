'''
Created on Apr 19, 2019

@author: dsj529
'''
from sklearn import datasets
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


dataset = datasets.load_digits()

image_features = dataset.images.reshape((len(dataset.images), -1))
image_targets = dataset.target

random_forest_model = RandomForestClassifier(n_jobs=-1,max_features='sqrt')

feature_train, feature_test, target_train, target_test = train_test_split(image_features, image_targets, test_size=.2)

param_grid = {
    "n_estimators" : [10,100,500,1000],
    "max_depth" : [1,5,10,15],
    "min_samples_leaf" : [1,2,3,4,5,10,15,20,30,40,50]              
}

grid_search = GridSearchCV(estimator=random_forest_model, param_grid=param_grid, cv=10)
grid_search.fit(feature_train, target_train)
print(grid_search.best_params_)
preds = grid_search.predict(feature_test)
print(classification_report(target_test, preds))


optimal_estimators = grid_search.best_params_.get("n_estimators")
optimal_depth = grid_search.best_params_.get("max_depth")
optimal_leaf = grid_search.best_params_.get("min_samples_leaf")