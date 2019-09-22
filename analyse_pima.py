import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
import os
from compare_classifiers import comparison_plot
from classifiers import svm_classifier, dtree_classifier, boost_classifier, knn_classifier, neural_net, tuned_boosting_classifier, tuned_neural_net, tuned_dtree_classifier, tuned_svm_classifier, tuned_knn_classifier, tuned_weighted_knn_classifier

warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 
os.system('clear')
plt.style.use('seaborn-whitegrid')

dataset = pd.read_csv('datasets/pima-indians-diabetes.data.csv')

dataset.columns = [
    "NumTimesPrg", "PlGlcConc", "BloodP",
    "SkinThick", "TwoHourSerIns", "BMI",
    "DiPedFunc", "Age", "HasDiabetes"] 


dataset.hist(bins=50, figsize=(20, 15))
plt.show()

# Calculate the median value for BMI
median_bmi = dataset['BMI'].median()
# Substitute it in the BMI column of the
# dataset where values are 0
dataset['BMI'] = dataset['BMI'].replace(to_replace=0, value=median_bmi)
# Calculate the median value for BloodP
median_bloodp = dataset['BloodP'].median()
# Substitute it in the BloodP column of the
# dataset where values are 0
dataset['BloodP'] = dataset['BloodP'].replace(to_replace=0, value=median_bloodp)
# Calculate the median value for PlGlcConc
median_plglcconc = dataset['PlGlcConc'].median()
# Substitute it in the PlGlcConc column of the
# dataset where values are 0
dataset['PlGlcConc'] = dataset['PlGlcConc'].replace(to_replace=0, value=median_plglcconc)
# Calculate the median value for SkinThick
median_skinthick = dataset['SkinThick'].median()
# Substitute it in the SkinThick column of the
# dataset where values are 0
dataset['SkinThick'] = dataset['SkinThick'].replace(to_replace=0, value=median_skinthick)
# Calculate the median value for TwoHourSerIns
median_twohourserins = dataset['TwoHourSerIns'].median()
# Substitute it in the TwoHourSerIns column of the
# dataset where values are 0
dataset['TwoHourSerIns'] = dataset['TwoHourSerIns'].replace(to_replace=0, value=median_twohourserins)


x = dataset.iloc[:,:-1].values
y = dataset.iloc[:, -1].values

fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(8,14))
fig.suptitle('Value histograms', fontsize=10)
ax = ax.flatten()

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)


print("X_train shape: {}".format(X_train.shape))
print("X_test shape: {}".format(X_test.shape))
print("y_train shape: {}".format(y_train.shape))
print("y_test shape: {}".format(y_test.shape))

scaller = StandardScaler()
X_train = scaller.fit_transform(X_train)
X_test = scaller.transform(X_test)

# Apply different machine learning classifiers

print('Apply different machine learning classifiers without Tuning')

svm_classifier(X_train,y_train,X_test,y_test,'linear','auto',plotting=False)

dtree_classifier(X_train,y_train,X_test,y_test,plotting=False)

knn_classifier(X_train,y_train,X_test,y_test, neighbors=1, plotting=False)

boost_classifier(X_train,y_train,X_test,y_test,plotting=False)

neural_net(X_train,y_train,X_test,y_test,learning_rate=1e-01,plotting=False)


print('Apply different machine learning classifiers with Tuning')

best_svm = tuned_svm_classifier(X_train,y_train,X_test,y_test,'linear','auto',plotting=False)

best_dtree = tuned_dtree_classifier(X_train,y_train,X_test,y_test,plotting=False)

best_knn = tuned_knn_classifier(X_train,y_train,X_test,y_test,neighbors=2, plotting=False)

best_knn_weighted = tuned_weighted_knn_classifier(X_train,y_train,X_test,y_test,neighbors=2, plotting=False)

best_boost = tuned_boosting_classifier(X_train,y_train,X_test,y_test,plotting=False)

best_nn = tuned_neural_net(X_train,y_train,X_test,y_test,learning_rate=0.0001,plotting=True)

comparison_plot(x,y,best_knn['n_neighbors'],'auto',best_svm['C'],best_dtree['max_depth'],best_nn['alpha'],best_boost['n_estimators'])
