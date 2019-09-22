import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
import os
import seaborn as sns
from compare_classifiers import comparison_plot
from classifiers import svm_classifier, dtree_classifier, boost_classifier, knn_classifier, neural_net, tuned_boosting_classifier, tuned_neural_net, tuned_dtree_classifier, tuned_svm_classifier, tuned_knn_classifier, tuned_weighted_knn_classifier

warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 
os.system('clear')
plt.style.use('seaborn-whitegrid')

data = pd.read_csv('datasets/data_banknote_authentication.csv')

fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(8,14))
fig.suptitle('Value histograms', fontsize=10)
ax = ax.flatten()

for i, col in enumerate(list(data)):
    ax[i].hist(x=data[col].astype(int))
    ax[i].set_xlabel(col)

plt.show(fig)

x = data.iloc[:,:-1].values
y = data.iloc[:, -1].values

fig = sns.countplot('Class', data=data)



X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

print("X_train shape: {}".format(X_train.shape))
print("X_test shape: {}".format(X_test.shape))
print("y_train shape: {}".format(y_train.shape))
print("y_test shape: {}".format(y_test.shape))

scaller = StandardScaler()
X_train = scaller.fit_transform(X_train)
X_test = scaller.transform(X_test)

# Apply different machine learning classifiers

svm_classifier(X_train,y_train,X_test,y_test,'linear','auto',plotting=False)

dtree_classifier(X_train,y_train,X_test,y_test,plotting=False)

boost_classifier(X_train,y_train,X_test,y_test,plotting=False)

knn_classifier(X_train,y_train,X_test,y_test, neighbors=1, plotting=False)

neural_net(X_train,y_train,X_test,y_test,learning_rate=1e-01,plotting=False)


print('********************************')

best_svm = tuned_svm_classifier(X_train,y_train,X_test,y_test,'linear','auto',plotting=False)

best_dtree = tuned_dtree_classifier(X_train,y_train,X_test,y_test,plotting=False)

best_knn = tuned_knn_classifier(X_train,y_train,X_test,y_test,neighbors=2, plotting=False)

best_knn_weighted = tuned_weighted_knn_classifier(X_train,y_train,X_test,y_test,neighbors=2, plotting=False)

best_boost = tuned_boosting_classifier(X_train,y_train,X_test,y_test,plotting=False)

best_nn = tuned_neural_net(X_train,y_train,X_test,y_test,learning_rate=0.0001,plotting=True)

comparison_plot(x,y,best_knn['n_neighbors'],'auto',best_svm['C'],best_dtree['max_depth'],best_nn['alpha'],best_boost['n_estimators'])

