# サンプルデータを使った機械学習

今回はskylearnのサンプルデータload_breast_cancerの癌の診断結果を元に、癌の分類をk近傍法を用いて学習させる。

プログラム本体はmain.pyに記載

## 使用したライブラリ

```
import pandas as pd #データをdataframe型に変換するため
from sklearn.datasets import load_breast_cancer #サンプルデータ
from sklearn.model_selection import train_test_split #サンプルデータ分割用
from sklearn.neighbors import KNeighborsClassifier #k近傍法の学習用
import matplotlib.pyplot as plt #グラフ表示用
```

## 変数の定義

```
MIN_NEIGHBOR = 1 #最小パラメータ
MAX_NEIGHBOR = 50 #最大パラメータ

correct = 0 #正解数 right
false = 0 #不正解数 wrong
max_acc = 0 #最大正答率
max_point = 0 #最高パラメータ
result = [] #グラフに表示するため正答率を格納
```

## データの準備

```
cancer_dataset = load_breast_cancer() #癌の診断結果を呼び出す

X_train, X_test, y_train, y_test = train_test_split(cancer_dataset['data'], cancer_dataset['target'], random_state=5) #学習用データと標本用データに分割

iris_dataframe = pd.DataFrame(X_train, columns=cancer_dataset.feature_names) #datafrme型に変換
```

## kパラメータのチューニング

```
for neighbor in range(MIN_NEIGHBOR, MAX_NEIGHBOR):
    knn = KNeighborsClassifier(n_neighbors=neighbor)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    for i,j in zip(y_test,y_pred):
        if i == j:
            correct += 1
            # print("真:",cancer_dataset['target_names'][i],',予測:', cancer_dataset['target_names'][j])
        else:
            false += 1
    acc = round(100 * correct / (correct + false), 3)
    print("k=" + str(neighbor) + ", 正答率 : " + str(acc) + "%")
    result.append(acc)
    if acc > max_acc:
        max_acc = acc
        max_point = neighbor
```

cancer 最高値はk=28の時で正答率は96.054%です

<a href="https://github.com/rikumomo0407/AI-Study-Record//raw/main/001/knn_cancer.png">
  <img width="50%" src="https://github.com/rikumomo0407/AI-Study-Record//raw/main/001/knn_cancer.png" />
</a>

iris 最高値はk=29の時で正答率は97.822%です

<a href="https://github.com/rikumomo0407/AI-Study-Record//raw/main/001/knn_iris.png">
  <img width="50%" src="https://github.com/rikumomo0407/AI-Study-Record//raw/main/001/knn_iris.png" />
</a>

wine 最高値はk=1の時で正答率は75.556%です

<a href="https://github.com/rikumomo0407/AI-Study-Record//raw/main/001/knn_wine.png">
  <img width="50%" src="https://github.com/rikumomo0407/AI-Study-Record//raw/main/001/knn_wine.png" />
</a>
