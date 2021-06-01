import streamlit as st
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

st.title("Classifiers")

st.write('''
# Explore multiple Classifier
Which is the best
''')

dataset_name = st.sidebar.selectbox("Select Dataset",("Iris","Breast Cancer", "Wine"))

classifer_name = st.sidebar.selectbox("Select Classifier",("KNN","SVM","Random Forest"))

def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
         data = datasets.load_breast_cancer()
    elif dataset_name == "Wine":
        data = datasets.load_wine()
    X = data.data
    y = data.target
    return X,y

X,y = get_dataset(dataset_name)
st.write("shape of dataset", X.shape)
st.write("number of classes",len(np.unique(y)))

def add_parameter_ui(clf_name):
    param =dict()
    if clf_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        param["K"] = K
    elif clf_name == "SVM":
        C = st.sidebar.slider("C", 1,15)
        param["C"] = C
    elif clf_name == "Random Forest":
        n_estimators = st.sidebar.slider("n_estimators", 10, 100)
        max_depth = st.sidebar.slider("max_depth", 1,10)
        param["n_estimators"] = n_estimators
        param["max_depth"] = max_depth
    return param

param = add_parameter_ui(classifer_name)

def get_classifier(clf_name,param):
    if clf_name == "KNN":
       clf = KNeighborsClassifier(n_neighbors=param["K"])
    elif clf_name == "SVM":
        clf = SVC(C=param["C"])
    elif clf_name == "Random Forest":
        clf = RandomForestClassifier(n_estimators=param["n_estimators"],max_depth=param["max_depth"], random_state=42)
    return clf

clf = get_classifier(classifer_name,param)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf.fit(x_train,y_train)

y_pred = clf.predict(x_test)

acc = accuracy_score(y_test,y_pred)
st.write(f"classifier name {classifer_name}")
st.write(f"accuracy is : {acc}")

pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:,0]
x2 = X_projected[:,1]

st.set_option('deprecation.showPyplotGlobalUse', False)
fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap="viridis")
plt.xlabel("Principle Component 1")
plt.ylabel("Principle Component 2")
plt.colorbar()
st.pyplot()