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

def preprocess(data):
    renamed = data.rename(index=str, columns={"column_a": "age", "column_b": "sex", "column_c": "chest_pain",
                          "column_d": "trestbps", "column_e": "cholesterol", "column_f": "fasting_bsugar", 
                          "column_g": "restecg", "column_h": "thalach", "column_i": "exercise_angina", 
                          "column_j": "oldpeak", "column_k": "slope", "column_l": "colored_vessels",
                          "column_m": "thal", "column_n": "target"})
    renamed.target = renamed.target.replace(to_replace=[2,3,4], value=1)    
    return renamed


raw_data = pd.read_csv('datasets\processed_cleveland_data.csv')
print(raw_data.info())
raw_data.head()

data = preprocess(raw_data)
data = data.dropna(subset=['thal', 'colored_vessels'])

print(data.info())
data.describe()

fig, ax = plt.subplots(ncols=3, nrows=5, figsize=(16,20))
fig.suptitle('Value histograms', fontsize=20)
ax = ax.flatten()

for i, col in enumerate(list(data)):
    ax[i].hist(x=data[col].astype(int))
    ax[i].set_xlabel(col)

plt.show(fig)

x = data.loc[:, :'thal'].values
y = data.loc[:, 'target'].values

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)

scaller = StandardScaler()
X_train = scaller.fit_transform(X_train)
X_test = scaller.transform(X_test)

# Apply different machine learning classifiers


dtree_classifier(X_train,y_train,X_test,y_test,plotting=False)


svm_classifier(X_train,y_train,X_test,y_test,'linear','auto',plotting=False)

boost_classifier(X_train,y_train,X_test,y_test,plotting=False)

knn_classifier(X_train,y_train,X_test,y_test, neighbors=2, plotting=False)

neural_net(X_train,y_train,X_test,y_test,learning_rate=1e-01,plotting=False)


print('********************************')
best_dtree = tuned_dtree_classifier(X_train,y_train,X_test,y_test,plotting=False)

best_svm = tuned_svm_classifier(X_train,y_train,X_test,y_test,'linear','auto',plotting=False)

best_knn = tuned_knn_classifier(X_train,y_train,X_test,y_test,neighbors=2, plotting=False)

best_knn_weighted = tuned_weighted_knn_classifier(X_train,y_train,X_test,y_test,neighbors=2, plotting=False)

best_boost = tuned_boosting_classifier(X_train,y_train,X_test,y_test,plotting=False)

best_nn = tuned_neural_net(X_train,y_train,X_test,y_test,learning_rate=0.01,plotting=True)

comparison_plot(x,y,best_knn['n_neighbors'],'auto',best_svm['C'],best_dtree['max_depth'],best_nn['alpha'],best_boost['n_estimators'])
