from logging import critical
import pandas as pd
import scipy.stats as stats
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.tree import plot_tree
from sklearn.tree import export_graphviz
from sklearn.tree import export_graphviz
from os import system
from graphviz import Source


st.title('DM Assignment-3')

uploaded_file = st.file_uploader(label="Choose a file", type=['csv', 'xlsx'])


global df
global numeric_columns
if uploaded_file is not None:
    print(uploaded_file)
    print("Hello")
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        print(e)
        df = pd.read_excel(uploaded_file)

try:
    st.write(df)
    numeric_columns = list(df.columns)
except Exception as e:
    print(e)
    st.write("Please upload file")

print("Dataset Length: ", len(df))
print("Dataset Shape: ", df.shape)

# Printing the dataset obseravtions
print("Dataset: ", df.head())

# Separate the independent and dependent variables using the slicing method.
# x = df.values[:, 1.5]
# y = df.values[:, 0]

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)

print(df.isnull().sum())  # checked if any attribute has null value


columns = df.columns
feature_cols = columns[1:]

target_cols = columns[0:1]

feature_data = df[:][1:]
target_data = df[:][0:1]
X = df[feature_cols]
Y = df[target_cols]

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=1)

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(max_depth=2, random_state=1)

# Train Decision Tree Classifer
clf = clf.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)


# text.delete("1.0","end")
st.write("Model Accuracy: " + str(metrics.accuracy_score(y_test, y_pred)))

c_matrix = confusion_matrix(y_test, y_pred)
print(c_matrix)
st.write(str(c_matrix))

print("True Positive:" + str(c_matrix[0][0]))
st.write("True Positive:" + str(c_matrix[0][0]))

print(X.columns)
print(Y.columns)
plt.figure(figsize=(4, 3), dpi=150)
# st.plotly_chart(plot_tree(clf, feature_names=X.columns, filled=True))
# st.write(plt.show(block=True))

# st.pyplot(tree.plot_tree(clf))
tree = export_graphviz(clf)

st.graphviz_chart(tree)

# Tabulate the results in confusion matrix and evaluate the performance of above classifier using following metrics :
st.write('Tabulate the results in confusion matrix and evaluate the performance of above classifier using following metrics :')

# tp = c_matrix[0][0]
# tn =
# fp =
# fn =


# Recognition rate

print('he;;p')
# precision score
val = metrics.precision_score(y_test, y_pred, average='macro')
print('Precision score : ' + str(val))
st.write('Precision score : ' + str(val))


# Accuracy score
val = metrics.accuracy_score(y_test, y_pred, average='macro')
st.write('Accuracy score : ' + str(val))

#
