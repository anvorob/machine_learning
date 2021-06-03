from pandas import read_csv,read_excel
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest,f_classif
from numpy import set_printoptions

# load data
filename = 'Mortgage.csv'
# filename1 = 'Iris.xlsx'
# filename2 = 'breast-cancer.xlsx'
names = ['Age','Ed','employ','address','income','debtinc','creddebt','othdebt','outcome']
# names1 = ['petal length','petal width','sepal length','sepal width','class']
# names2 = ['age','menopause','tumor-size','inv-nodes','node-caps','deg-malig','breast','breast-quad','irradiat','class']
dataframe = read_csv(filename, names=names,skiprows=[0])
dataframe = dataframe.head(700);
nrow, ncol = dataframe.shape;

# Question a.
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]

# feature extraction
test = SelectKBest(score_func=f_classif, k=4)
fit = test.fit(X, Y)
# summarize scores
set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X)
plt.bar(names[:-1],fit.scores_)
plt.show()


#Question b.
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

predictors = dataframe.iloc[:,:ncol-1]
target = dataframe.iloc[:,-1]


pred_train, pred_test, tar_train, tar_test  = train_test_split(predictors, target,  stratify= target, test_size=.3,random_state=1)
classifier = DecisionTreeClassifier()
classifier = classifier.fit(pred_train, tar_train)
scores = cross_val_score(classifier, pred_train, tar_train, cv=10)  # taining set is used for cross validation
print("Accuracy score under cross validation default:", scores.mean()," Depth: ", classifier.get_depth())

def buildTree(criterion):
    classifier = DecisionTreeClassifier( criterion=criterion)
    classifier = classifier.fit(pred_train, tar_train)
    scores = cross_val_score(classifier, pred_train, tar_train, cv=10)  # taining set is used for cross validation
    print("Accuracy score for Criterion variable under cross validation :", scores.mean()," Depth: ", classifier.get_depth())

for i in range(5,25):
    buildTree("entropy");

def buildTreeSplLfN(max_leaf_nodes):
    classifier = DecisionTreeClassifier( max_leaf_nodes=max_leaf_nodes)
    classifier = classifier.fit(pred_train, tar_train)
    scores = cross_val_score(classifier, pred_train, tar_train, cv=10)  # taining set is used for cross validation
    print("Accuracy score for max_leaf_nodes under cross validation :", scores.mean()," Depth: ", classifier.get_depth())
print()
for i in range(5,40):
    buildTreeSplLfN( i);

#   Question d
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(classifier, pred_test, tar_test)
plt.show()


