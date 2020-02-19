import pandas as pd
import numpy as np
from sklearn.datasets import load_wine

wine = load_wine()

data = pd.DataFrame(data= np.c_[wine['data'], wine['target']],
                     columns= wine['feature_names'] + ['target'])
data.head()

data.info()
data.describe()
print(data.columns)

import matplotlib.pyplot as plt
import seaborn as sns

# calculating the Correlation Matrix
corr = data.corr()
print(corr)
# plotting the heatmap
#Heatmap makes it easy to identify which features are most related to the target variable,
# we will plot heatmap of correlated features using the seaborn library.
fig = plt.figure(figsize=(5,4))
sns.heatmap(corr,
        xticklabels=corr.columns,
        yticklabels=corr.columns,
            linewidths=.75)
plt.show()

#Feature Importance
# from sklearn.ensemble import ExtraTreesClassifier
#
# model = ExtraTreesClassifier()
# model.fit(X,Y)
#
# print(model.feature_importances_)
#
# #Visualization
# features_importances = pd.Series(model.feature_importances_, index=X.columns)
# features_importances.nlargest(12).plot(kind='barh')
# plt.show()

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X = data.iloc[:, :-1]
y = data.iloc[:, -1]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


clf = LogisticRegression()
clf = clf.fit(X_train,y_train)

clf

from sklearn.metrics import accuracy_score
import pickle

y_pred = clf.predict(X_test)

print("accuracy_score: %.2f"
      % accuracy_score(y_test, y_pred))

path='./lib/models/LogisticRegression.pkl'

with open(path, 'wb') as file:
    pickle.dump(clf, file)