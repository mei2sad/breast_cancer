#####svm########################
import sklearn.datasets as d
import pandas as pd
import sklearn.model_selection as ms
import matplotlib.pyplot as plt
import sklearn.svm as svm
import sklearn.metrics as mt
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

x = d.load_breast_cancer()
cancer = pd.DataFrame(data = x.data, columns = x.feature_names)
cancer['target'] = x.target

X = cancer.iloc[:,:-1]
y = cancer.iloc[:,-1]

X = cancer.iloc[:,:-1]
y = cancer.iloc[:,-1]

scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
X_train, X_test, y_train, y_test = ms.train_test_split(X_scaled, y, test_size = 0.3, random_state = 42)

svm_clf =svm.SVC(kernel = 'linear', random_state=42)

scores = cross_val_score(svm_clf, X_scaled, y, cv = 5)
print(scores)

print('교차검증 평균%.4f '% scores.mean())

parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 25, 50, 100],'gamma':[0.1, 0.25, 0.5, 1]}

grid_svm = GridSearchCV(svm_clf, parameters, cv = 5)

grid_svm.fit(X_train, y_train)

result = pd.DataFrame(grid_svm.cv_results_['params'])
result['mean_test_score'] = grid_svm.cv_results_['mean_test_score']
result.sort_values(by='mean_test_score', ascending=False)

print("그리드 서치를 통한 하이퍼파라미터 최적값 서치:",end="")
print(grid_svm.best_params_)
print("하이퍼 파라미터 조정 후 SVM: %.4f"%grid_svm.best_score_)
