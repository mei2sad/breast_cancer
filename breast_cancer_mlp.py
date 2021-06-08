
# coding: utf-8

from sklearn.datasets import load_breast_cancer

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, classification_report
from sklearn.preprocessing import Normalizer, MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, LabelEncoder
from sklearn.pipeline import Pipeline



cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data,cancer.target, random_state=42, test_size=0.2)



# 훈련 세트 각 특성의 평균

mean_on_train = X_train.mean(axis=0)

# 훈련 세트 각 특성의 표준 편차

std_on_train = X_train.std(axis=0)



# 표준 정규분포

X_train_scaled = (X_train-mean_on_train)/std_on_train

X_test_scaled = (X_test-mean_on_train)/std_on_train


/*
random_state = 42
mlp_activation = ['identity', 'logistic', 'tanh', 'relu']
mlp_solver = ['lbfgs', 'sgd', 'adam']
mlp_max_iter = range(100, 1000, 10000)
mlp_alpha = [1e-4, 1e-3, 0.01, 0.1, 1]
preprocess = [Normalizer(), MinMaxScaler(), StandardScaler(), RobustScaler(), QuantileTransformer()]

mlp_param_grid = [
    {
        'preprocess': preprocess,
        'classification__activation': mlp_activation,
        'classification__solver': mlp_solver,
        'classification__random_state': [random_state],
        'classification__max_iter': mlp_max_iter,
        'classification__alpha': mlp_alpha
    }
]

pipe = Pipeline(steps=[
    ('preprocess', StandardScaler()),
    ('classification', MLPClassifier())
])

strat_k_fold = StratifiedKFold(
     n_splits=10,
     random_state=42,
     shuffle=True
)

mlp_grid = GridSearchCV(
     pipe,
     param_grid=mlp_param_grid,
     cv=strat_k_fold,
     scoring='f1',
     n_jobs=-1,
     verbose=2
)

mlp_grid.fit(cancer.data, cancer.target)

# Best MLPClassifier parameters
print(mlp_grid.best_params_)
# Best score for MLPClassifier with best parameters
print('\nBest F1 score for MLP: {:.2f}%'.format(mlp_grid.best_score_ * 100))

best_params = mlp_grid.best_params_
*/
###############위의 코드만 실행시 그리드 서치로 최적값 찾기 가능############################

###############아래 코드는 위에서 찾은 최적의 하이퍼 파라미터를 대입하여 실행시킨 코드######################


scaler = StandardScaler()

print('\nData preprocessing with {scaler}\n'.format(scaler=scaler))

X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)

mlp = MLPClassifier(
    max_iter=100,
    alpha=0.01,
    activation='logistic',
    solver='adam',
    random_state=42
)
mlp.fit(X_train_scaler, y_train)

mlp_predict = mlp.predict(X_test_scaler)
mlp_predict_proba = mlp.predict_proba(X_test_scaler)[:, 1]

print('MLP Accuracy: {:.2f}%'.format(accuracy_score(y_test, mlp_predict) * 100))
print('MLP AUC: {:.2f}%'.format(roc_auc_score(y_test, mlp_predict_proba) * 100))
print('MLP Classification report:\n\n', classification_report(y_test, mlp_predict))
print('MLP Training set score: {:.2f}%'.format(mlp.score(X_train_scaler, y_train) * 100))
print('MLP Testing set score: {:.2f}%'.format(mlp.score(X_test_scaler, y_test) * 100))
