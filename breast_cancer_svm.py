import sklearn.datasets as d
import pandas as pd
import sklearn.model_selection as ms
import matplotlib.pyplot as plt
# breast_cancer 데이터 셋 로드
x = d.load_breast_cancer()
cancer = pd.DataFrame(data = x.data, columns = x.feature_names)
cancer['target'] = x.target

X = cancer.iloc[:,:-1]
y = cancer.iloc[:,-1]

X = cancer.iloc[:,:-1]
y = cancer.iloc[:,-1]

from sklearn.preprocessing import StandardScaler

# StandarScaler 적용

scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

# 변환된 X로 데이터 분할
X_train, X_test, y_train, y_test = ms.train_test_split(X_scaled, y, 
                                                      test_size = 0.3, random_state = 42)

import sklearn.svm as svm
import sklearn.metrics as mt
from sklearn.model_selection import cross_val_score, cross_validate

# SVM, kernel = 'linear'로 선형분리 진행
 
svm_clf =svm.SVC(kernel = 'linear', random_state=42)

# 변환된 X로 교차검증

scores = cross_val_score(svm_clf, X_scaled, y, cv = 5)
print(scores)

print('교차검증 평균%.4f '% scores.mean())
