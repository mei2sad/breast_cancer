from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

breast_cancer_data = load_breast_cancer()
k=1
#k값 0이면 실행이 안됨

df_data = pd.DataFrame(breast_cancer_data.data)
df_labels = pd.DataFrame(breast_cancer_data.target)


def min_max_normalize(lst):
    

    normalized = []
    
    for value in lst:
        normalized_num = (value - min(lst)) / (max(lst) - min(lst))
        normalized.append(normalized_num)
    
    return normalized

for x in range(len(df_data.columns)):
    df_data[x] = min_max_normalize(df_data[x])


training_data, validation_data , training_labels, validation_labels = train_test_split(df_data, df_labels, test_size = 0.2, random_state = 42)

import matplotlib.pyplot as plt

k_list = range(1,101)
accuracies = []

for k in k_list:
  classifier = KNeighborsClassifier(n_neighbors = k)
  classifier.fit(training_data, training_labels)
  accuracies.append(classifier.score(validation_data, validation_labels))

plt.plot(k_list, accuracies)
plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.title("Breast Cancer Classifier Accuracy")
plt.show()

#k값을 1~101까지 반복 후 그래프를 보면 1~20이후는 급격히 떨어짐, 따라서 1~20 사이 값을 자세히 추출

k_list = range(1,20)
accuracies = []

for k in k_list:
  classifier = KNeighborsClassifier(n_neighbors = k)
  classifier.fit(training_data, training_labels)
  accuracies.append(classifier.score(validation_data, validation_labels))

plt.plot(k_list, accuracies)
plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.title("Breast Cancer Classifier Accuracy")
plt.show()

#정확도 출력 부분

for i in range(1,20):
    classifier = KNeighborsClassifier(n_neighbors = i)
    classifier.fit(training_data, training_labels.values.ravel())
    #print(classifier.score(validation_data, validation_labels))
    print("K=%d"%i)
    print('KNN Accuracy: {:.2f}%'.format(classifier.score(validation_data, validation_labels) * 100))
    
    
    #########################주석 수정된 파일#################################
    ###############1.02 테스트용 주석##########################
