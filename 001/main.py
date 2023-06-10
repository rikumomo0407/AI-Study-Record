import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

correct = 0
false = 0
max_acc = 0
max_point = 0
neighbors = 1
max_neighbor = 50
result = []

cancer_dataset = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer_dataset['data'], cancer_dataset['target'], random_state=5)

iris_dataframe = pd.DataFrame(X_train, columns=cancer_dataset.feature_names)
print(iris_dataframe)

for i in range(max_neighbor):
    knn = KNeighborsClassifier(n_neighbors=neighbors)
    # knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    for i,j in zip(y_test,y_pred):
        if i == j:
            correct += 1
            # print("真:",cancer_dataset['target_names'][i],',予測:', cancer_dataset['target_names'][j])
        else:
            false += 1
    acc = round(100 * correct / (correct + false), 3)
    print("k=" + str(neighbors) + ", 正答率 : " + str(acc) + "%")
    result.append(acc)
    if acc > max_acc:
        max_acc = acc
        max_point = neighbors
    neighbors += 1

print("最高値はk=" + str(max_point) + "の時で正答率は" + str(max_acc) + "%です")
plt.plot(list(range(1, max_neighbor + 1)), result)
plt.show()
