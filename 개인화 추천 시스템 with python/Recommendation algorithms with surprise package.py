!pip install scikit-surprise

from surprise import BaselineOnly
from surprise import KNNWithMeans
from surprise import SVD
from surprise import SVDpp
from surprise import Dataset
from surprise import accuracy
from surprise import Reader
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split
from surprise.model_selection import GridSearchCV

import numpy as np
import matplotlib.pyplot as plt

data = Dataset.load_builtin('ml-100k')
train_set, test_set = train_test_split(data, test_size = 0.25)

# 알고리즘 별 정확도 계산
algorithms = [BaselineOnly, KNNWithMeans, SVD, SVDpp]
names = []
results = []
for option in algorithms:
  algo = option()
  names.append(option.__name__)
  algo.fit(train_set)
  predictions = algo.test(test_set)
  results.append(accuracy.rmse(predictions))

names = np.array(names)
results = np.array(results)

index = np.argsort(results)
plt.plot(names[index], results[index])
plt.ylim(0.8, 1)
plt.show()

# 알고리즘 옵션 변경, 정확도 계산
sim_options = {'names': 'pearson_baseline', 'user_based': True}
algo = KNNWithMeans(k=30, sim_options=sim_options)
algo.fit(train_set)
predictions = algo.test(test_set)
accuracy.rmse(predictions)

# KNN 격자 탐색 시행
param_grid = {'k': [5, 10, 15, 25], 'sim_options': {'name': ['pearson_baseline', 'cosine'], 'user_based': [True, False]}}
gs = GridSearchCV(KNNWithMeans, param_grid, measures = ['rmse'], cv = 4)
gs.fit(data)

print(gs.best_score['rmse'])
print(gs.best_params['rmse'])

# SVD 격자 탐색 시행
param_grid = {
    'n_epochs': [70, 80, 90],
    'lr_all': [0.005, 0.006, 0.007],
    'reg_all': [0.05, 0.07, 0.1]
}
gs = GridSearchCV(svd, param_grid, measures=['rmse'], cv=4)
gs.fit(data)

print(gs.best_score['rmse'])
print(gs.best_params['rmse'])
