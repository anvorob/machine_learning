import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest,chi2,mutual_info_regression
from numpy import set_printoptions
from scipy.io import arff
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB

columns = ['age','gender','chest pain type','rest bp','cholestoral','blood sugar','ECG results','max hr','CP aft wrkout','peak hr aft wrko','hr variation','status of bld ves','bld supply st.','class']
path="Heart.arff"
data = arff.loadarff(path)
df = pd.DataFrame(data[0])

#Encoder
from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()
df['class'] = class_le.fit_transform(df['class'].values)



#Question a. feature extraction
array = df.values
X = array[:,0:13]
Y = array[:,13]
test = SelectKBest(score_func=mutual_info_regression, k=4)
fit = test.fit(X, Y)
# summarize scores
set_printoptions(precision=3)
# print(fit.scores_)
features = fit.transform(X)
# summarize selected features
# print(features[0:5,:])
plt.bar(columns[:-1],fit.scores_)
plt.show()


nrow, ncol = df.shape;
predictors = df.iloc[:,:ncol-1]
#index to last column to obtain class values
target = df.iloc[:,-1]

# print ("\n correlation Matrix")
corrMatrix = predictors.corr()
import seaborn as sn
import matplotlib.pyplot as plt
sn.heatmap(corrMatrix, annot=True)
plt.show()


#Question b
from sklearn.metrics import  accuracy_score
pred_features = predictors[['status of blood vessels', 'peak heart rate after exercise', 'blood supply status']]
# print(pred_features);
pred_train, pred_test, tar_train, tar_test  = train_test_split(predictors, target,  stratify= target, test_size=.3, random_state=1)

######----------
test = SelectKBest(score_func=mutual_info_regression, k=3)
fit = test.fit(pred_train, tar_train)
# summarize scores
set_printoptions(precision=3)
# print(fit.scores_)
features = fit.transform(pred_train)
# summarize selected features
# print(features[0:5,:])
plt.bar(columns[:-1],fit.scores_)
plt.show()
########---------------


gnb = GaussianNB()
gnb= gnb.fit(pred_train, tar_train)
predictions = gnb.predict(pred_test)
sh = pred_test.shape
print("Accuracy score of our model with Gaussian Naive Bayes:", accuracy_score(tar_test, predictions))



#Question C
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier = classifier.fit(pred_train, tar_train)
predictions = classifier.predict(pred_test)
prob = classifier.predict_proba(pred_test)
scores = cross_val_score(classifier, pred_train, tar_train, cv=10)  # taining set is used for cross validation
print("Accuracy score for DecisionTreeClassifier under cross validation :", scores.mean())
print(classifier.feature_importances_);
