# サンプルデータを使った機械学習

今回はskylearnのサンプルデータload_breast_cancerの癌の診断結果を元に、癌の分類を学習させる。

プログラム本体はmain.pyに記載

## 使用したライブラリ

'''
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
'''

## 変数の定義

'''
correct = 0 #正解数
false = 0 #不正解数
max_acc = 0 #最大正答率
max_point = 0 #最高パラメータ
neighbors = 1 #パラメータの初期値
max_neighbor = 50 #最大パラメータ
result = [] #グラフに表示するため正答率を格納
'''

## データの準備

'''
cancer_dataset = load_breast_cancer() #癌の診断結果を呼び出す

X_train, X_test, y_train, y_test = train_test_split(cancer_dataset['data'], cancer_dataset['target'], random_state=5) #学習用データと標本用データを分割

iris_dataframe = pd.DataFrame(X_train, columns=cancer_dataset.feature_names) #datafrme型に変換
print(iris_dataframe)
'''

## kパラメータのチューニング

'''
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
'''

