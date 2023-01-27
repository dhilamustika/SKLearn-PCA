from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.decomposition import PCA


iris = datasets.load_iris()
attribute = iris.data
label = iris.target

# split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    attribute, label, test_size=0.2, random_state=1)

# calculate model accuracy without PCA
decision_tree = tree.DecisionTreeClassifier()
model1 = decision_tree.fit(X_train, y_train)
model1.score(X_test, y_test)

# create a PCA object with 4 principal components
pca = PCA(n_components=4)

# apply PCA to the dataset
pca_attributes = pca.fit_transform(X_train)

# check the variance of each attribute
pca.explained_variance_ratio_

# PCA with 2 principal components
pca = PCA(n_components = 2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.fit_transform(X_test)

# test the accuracy of the classifier
model2 = decision_tree.fit(X_train_pca, y_train)
model2.score(X_test_pca, y_test)
