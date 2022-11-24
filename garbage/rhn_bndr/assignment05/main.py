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
from sklearn.tree import DecisionTreeRegressor


st.title('DM Assignment-5')

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


print(df.isnull().sum())  # checked if any attribute has null value


columns = df.columns
feature_cols = columns[4:]

target_cols = columns[0:4]

feature_data = df[:][1:]
target_data = df[:][0:1]
X = df[feature_cols]
Y = df[target_cols]
print('----')
print(X)
print('----')
print(Y)

# X_train, X_test, y_train, y_test = train_test_split(
#     X, Y, test_size=0.2, random_state=0)

# regressor = DecisionTreeRegressor()
# regressor.fit(X_train, y_train)

# # Predict the response for test dataset
# y_pred = regressor.predict(X_test)


# # text.delete("1.0","end")
# st.write("Model Accuracy: " + str(metrics.accuracy_score(y_test, y_pred)))

# c_matrix = confusion_matrix(y_test, y_pred)
# print(c_matrix)
# st.write(str(c_matrix))

# # st.pyplot(tree.plot_tree(clf))
# tree = export_graphviz(clf)

# st.graphviz_chart(tree)
