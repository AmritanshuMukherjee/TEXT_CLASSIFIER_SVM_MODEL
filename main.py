import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('/content/lr.csv')
X = dataset.iloc[:, [0,1]].values
y = dataset.iloc[:, 2].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.25, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from sklearn.svm import SVC
lr_model = SVC(gamma="auto")
lr_model.fit(X_train,y_train)
from sklearn import metrics
predictions = lr_model.predict(X_test)
print(metrics.classification_report(y_test,predictions))
