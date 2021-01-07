import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import boston_housing
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LinearRegression


(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

linear = LinearRegression()
pipe = make_pipeline(StandardScaler(), linear)
scores = cross_validate(pipe, train_data, train_targets, cv=10, return_train_score = True)
print(np.mean(scores['test_score']))
