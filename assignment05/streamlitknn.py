from cProfile import label
from collections import Counter
import math
import smtpd
from ssl import Options
from statistics import mean
from turtle import update

from Tools.scripts.make_ctype import values
from scipy.stats import pearsonr

import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import plotly.express as px
from logging import critical
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression

import scipy.stats as stats

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn import tree

# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.tree import plot_tree
from sklearn.tree import export_graphviz
from os import system
# from graphviz import Source
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model


# genre = st.radio(
#     "Main menu",
#     ('Assignment-1', 'Assignment-2', 'Assignment-3', 'Assignment-4', 'Assignment-5'))

genre='Assignment-5'

if genre == 'Assignment-1':
    st.title('DM Assignment-1')

    uploaded_file = st.file_uploader(label="Choose a file",
                                     type=['csv', 'xlsx'])

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

    st.subheader(
        "The measures of central tendency and dispersion of data")

    mean1 = df['sepal.length'].mean()

    median1 = df['sepal.length'].median()
    mode1 = df['sepal.length'].mode()
    # midRange1 = df['Salary'].()

    std1 = df['sepal.length'].std()
    var1 = df['sepal.length'].var()
    quantile1 = df['sepal.length'].quantile()

    print('Mean salary: ' + str(mean1))

    st.write("Mean : " + str(mean1))
    st.write("Median : " + str(median1))
    st.write("Mode : " + str(mode1))
    # st.write("Midrange : " + str(midRange1))
    st.write("variance : " + str(var1))
    st.write("Standard Deviation : " + str(std1))
    st.write("Quantile : " + str(quantile1))

    print(df.columns.values)
    print(df['sepal.length'].values)

    # Calculate mean
    mean1 = 0
    count = 0
    for x in df['sepal.length'].values:
        mean1 += x
        count = count + 1
    mean1 = mean1/count
    print(mean1)

    # Calcualte Median
    data1 = df['sepal.length'].values

    def median(data1):
        sorted_data = sorted(data1)
        data_len = len(sorted_data)

        middle = (data_len - 1) // 2

        if middle % 2:
            return sorted_data[middle]
        else:
            return (sorted_data[middle] + sorted_data[middle + 1]) / 2.0

    print(median(data1))

    n = len(data1)

    # calulate mode
    d = Counter(data1)
    get_mode = dict(d)
    mode = [k for k, v in get_mode.items() if v == max(list(d.values()))]

    if len(mode) == n:
        get_mode = "No mode found"
    else:
        get_mode = "Mode is / are: " + ', '.join(map(str, mode))

    print(get_mode)

    # calculate midrange
    n = len(data1)
    arr = []
    for i in range(len(data1)):
        # arr.append(df.loc[i, attribute])
        arr.sort()
    # print("Midrange of given dataset is ("+") "+str((arr[n-1]+arr[0])/2))
    # st.write("Midrange of given dataset is ("
    #  + ") "+str((arr[n-1]+arr[0])/2))

    def variance(data):
        # Number of observations
        n = len(data)
        # Mean of the data
        mean = sum(data) / n
        # Square deviations
        deviations = [(x - mean) ** 2 for x in data]
        # Variance
        variance = sum(deviations) / n
        return variance

    print(variance(data1))

    def stdev(data):
        var = variance(data)
        std_dev = math.sqrt(var)
        return std_dev

    print(stdev(data1))

    def rank(data1):
        sorted_data = sorted(data1)
        n = len(sorted_data)
        mid = 0
        if n % 2:
            mid = (n + 1)/2
        else:
            mid = n/2
        mid = int(mid)
        print('mid' + str(mid))
        print('Range' + str(n - mid))
        st.write("Range : " + str(n - mid))
        # print(sorted_data[mid])
        # iq1 = int(mid/2)
        # iq2 = iq1 + int((n-mid)/2)
        # r1 = sorted_data[iq2] - sorted_data[iq1]
        # print(str(iq1) + ' ' + str(iq2))
        # print(r1)
        Q1 = np.median(sorted_data[:mid])

        # Third quartile (Q3)
        Q3 = np.median(sorted_data[mid:])

        # Interquartile range (IQR)
        IQR = Q3 - Q1
        print('Interquartile Range' + str(IQR))
        st.write("Interquartile Range : " + str(IQR))

    # smtpd.qqplot(data1, line='45')
    # st.show(smtpd.qqplot(data1, line='45'))

    rank(data1)

    st.subheader(
        "Graphical display of above calculated statistical description of data")
    # Add a select weight to the sidebar
    chart_select = st.sidebar.selectbox(
        label="Select the chart type",
        options=['Scatterplits', 'Lineplots', 'Histogram', 'Boxplot']
    )

    if chart_select == 'Scatterplits':
        st.sidebar.subheader("Scatterplot Settings")
        try:
            x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
            y_values = st.sidebar.selectbox('Y axis', options=numeric_columns)
            plot = px.scatter(data_frame=df, x=x_values, y=y_values)
            # display chart
            st.plotly_chart(plot)
        except Exception as e:
            print(e)

    if chart_select == 'Lineplots':
        st.sidebar.subheader("Lineplot Settings")
        try:
            x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
            y_values = st.sidebar.selectbox('Y axis', options=numeric_columns)
            plot = px.line(data_frame=df, x=x_values, y=y_values)
            # display chart
            st.plotly_chart(plot)
        except Exception as e:
            print(e)

    if chart_select == 'Histogram':
        st.sidebar.subheader("Histogram Settings")
        try:
            x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
            y_values = st.sidebar.selectbox('Y axis', options=numeric_columns)
            plot = px.histogram(data_frame=df, x=x_values, y=y_values)
            # display chart
            st.plotly_chart(plot)
        except Exception as e:
            print(e)

    if chart_select == 'Boxplot':
        st.sidebar.subheader("Boxplot Settings")
        try:
            x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
            y_values = st.sidebar.selectbox('Y axis', options=numeric_columns)
            plot = px.box(data_frame=df, x=x_values, y=y_values)
            # display chart
            st.plotly_chart(plot)
        except Exception as e:
            print(e)

if genre == 'Assignment-2':
    st.title('DM Assignment-2')

    uploaded_file = st.file_uploader(
        label="Choose a file", type=['csv', 'xlsx'])

    if uploaded_file is not None:
        print(uploaded_file)
        df = pd.read_csv(uploaded_file)

        st.write(df)
        numeric_columns = list(df.columns)

        pd.set_option('display.max_columns', 100)
        edu = pd.read_csv("xAPI-Edu-Data.csv")

        # print(edu.columns)

        # contingency = pd.crosstab(edu['ParentAnsweringSurvey'], edu["GradeID"])

        # Add a select weight to the sidebar
        st.sidebar.subheader("Select Attributes")
        try:
            attribute1 = st.sidebar.selectbox(
                'Attribute-1', options=numeric_columns)
            attribute2 = st.sidebar.selectbox(
                'Attribute-2', options=numeric_columns)
            contingency = pd.crosstab(edu[attribute1], edu[attribute2])

            # display chart
            st.subheader("Contingency Table")
            st.table(contingency)
        except Exception as e:
            print(e)

        # Print contingency table
        print(contingency)

        # st.write(contingency)

        observed_values = contingency.values
        print(observed_values)

        #  we will get expected values
        val = stats.chi2_contingency(contingency)
        print(val)
        st.write(val)

        alpha = 0.05
        dof = 9  # degree of freedom

        # calculating critical value
        critical_value = stats.chi2.ppf(q=1 - alpha, df=dof)
        print(critical_value)

        # st.subheader("Chi-square value(Expected)" + str(val[''][]))
        st.subheader("Chi-square value(Critical) : " + str(critical_value))

        # Conslusion
        st.subheader("Conclusion : ")

        chi2_value_expected = val[0]
        print(chi2_value_expected)
        if chi2_value_expected < critical_value:
            print("So, the chi2 test statistic ( " + str(chi2_value_expected) + ") we calculated is smaller than the chi2 critical ( " +
                  str(critical_value) + ") value we have got from the distribution. So, we do not have enough evidence to reject the null hypothesis.")
            st.write("So, the chi2 test statistic ( " + str(chi2_value_expected) + ") we calculated is smaller than the chi2 critical ( " +
                     str(critical_value) + ") value we have got from the distribution. So, we do not have enough evidence to reject the null hypothesis.")
        else:
            print("So, the chi2 test statistic ( " + str(chi2_value_expected) + ") we calculated is greater than the chi2 critical ( " +
                  str(critical_value) + ") value we have got from the distribution. So, we do have enough evidence to reject the null hypothesis.")

        # Correlation analysis : Correlation coefficient (Pearson coefficient) & Covariance
        st.subheader(
            "Correlation analysis using Correlation coefficient (Pearson coefficient) & Covariance")

        st.sidebar.subheader("Select Attributes(for correlation coefficient)")
        try:
            attribute1 = st.sidebar.selectbox(
                'Attribute-11', options=numeric_columns)
            attribute2 = st.sidebar.selectbox(
                'Attribute-22', options=numeric_columns)
            var = np.corrcoef(df[attribute1], df[attribute2])
            corr, _ = pearsonr(df[attribute1], df[attribute2])
            print('Pearsons correlation: %.3f' % corr)
            st.write('Pearsons correlation: %.3f' % corr)
            # display chart
            print('helloone')
            st.subheader("Table")
            st.write(var)
        except Exception as e:
            print(e)

        # Calculating coefficient with formulas
        # list1 = list(newDf['salary'])
        # list2 = list(newDf['salbegin'])
        # n = len(list1)
        # mean1 = sum(list1) / n
        # mean2 = sum(list2) / n

        # n, mean1, mean2
        # num = 0
        # for i in range(n):
        #     num = num + (list1[i] - mean1) * (list2[i] - mean2)

        # num
        # SS1 = 0
        # for i in list1:
        #     SS1 = SS1 + (i - mean1)**2

        # SS2 = 0
        # for i in list2:
        #     SS2 = SS2 + (i - mean2)**2

        # SS1, SS2
        # pearsonCorr = num / (SS1 * SS2)**0.5
        # pearsonCorr

        # Normalization using following techniques :
        # 1. Min-max normalization
        attribute_to_be_normalized = st.sidebar.selectbox(
            'attribute_to_be_normalized', options=numeric_columns)
        min = df['attribute_to_be_normalized'].min()
        max = df[attribute_to_be_normalized].max()
        for i in range(len(df)):
            df.loc[i, attribute_to_be_normalized] = (
                (df.loc[i, attribute_to_be_normalized]-min)/(max-min))
            print(df['attribute_to_be_normalized'])
            st.write(df['attribute_to_be_normalized'])

        # st.plotly_chart(df.plot(kind='bar'))
        # x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
        # y_values = st.sidebar.selectbox('Y axis', options=numeric_columns)
        # plot = px.scatter(data_frame=df, x=x_values, y=y_values)
        # # display chart
        # st.plotly_chart(plot)
        # # copy the data
        # df_min_max_scaled = df.copy()

        # v = stats.zscore(df_min_max_scaled)

        # # view normalized data
        # st.plotly_chart(df_min_max_scaled)

        # rho = np.corrcoef(x)

        # fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 3))
        # for i in [0,1,2]:
        #     ax[i].scatter(x[0,],x[1+i,])
        #     ax[i].title.set_text('Correlation = ' + "{:.2f}".format(rho[0,i+1]))
        #     ax[i].set(xlabel='x',ylabel='y')
        # fig.subplots_adjust(wspace=.4)
        # plt.show()

if genre == 'Assignment-3':
    st.title('DM Assignment-3')

    uploaded_file = st.file_uploader(
        label="Choose a file", type=['csv', 'xlsx'])

    if uploaded_file is not None:
        print(uploaded_file)
        df = pd.read_csv(uploaded_file)

        st.write(df)
        numeric_columns = list(df.columns)

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
        st.write("Model Accuracy: " +
                 str(metrics.accuracy_score(y_test, y_pred)))

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

        tp = c_matrix[1][1]
        tn = c_matrix[2][2]
        fp = c_matrix[1][2]
        fn = c_matrix[2][1]

        # Recognition rate

        print('he;;p')
        # precision score
        val = metrics.precision_score(y_test, y_pred, average='macro')
        print('Precision score : ' + str(val))
        st.write('Precision score : ' + str(val))

        # Accuracy score
        val = metrics.accuracy_score(y_test, y_pred)
        st.write('Accuracy score : ' + str(val))

        #
if genre == 'Assignment-4':

    st.title('DM Assignment-4')

    uploaded_file = st.file_uploader(
        label="Choose a file", type=['csv', 'xlsx'])

    if uploaded_file is not None:
        print(uploaded_file)
        df = pd.read_csv(uploaded_file)

        st.write(df)
        numeric_columns = list(df.columns)

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
        st.write("Model Accuracy: " +
                 str(metrics.accuracy_score(y_test, y_pred)))

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

        # I believe that this answer is more correct than the other answers here:

        def tree_to_code(tree, feature_names):
            tree_ = tree.tree_
            feature_name = [
                feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
                for i in tree_.feature
            ]
            print("def tree({}):".format(", ".join(feature_names)))

            def recurse(node, depth):
                indent = "  " * depth
                if tree_.feature[node] != _tree.TREE_UNDEFINED:
                    name = feature_name[node]
                    threshold = tree_.threshold[node]
                    print("{}if {} <= {}:".format(indent, name, threshold))
                    recurse(tree_.children_left[node], depth + 1)
                    print("{}else:  # if {} > {}".format(
                        indent, name, threshold))
                    recurse(tree_.children_right[node], depth + 1)
                else:
                    print("{}return {}".format(indent, tree_.value[node]))

            recurse(0, 1)

if genre == "Assignment-5":
    st.title('Assignment NO. 05')

    uploaded_file = st.file_uploader(label="Choose a file",
                                     type=['csv', 'xlsx'])

    if uploaded_file is not None:
        print(uploaded_file)
        df = pd.read_csv(uploaded_file)
        st.dataframe(df)
        numeric_columns = list(df.columns)

        st.dataframe(df)

        rad5 = st.radio("Select", ["k-NN classifier","Regression classifier"])

        if rad5 == "Regression classifier":

            int_class = []

            # Setosa = 1
            # Versicolor = 2
            # Virginica = 3

            for i in df['variety']:
                if i == 'Setosa':
                    int_class.append(1)
                if i == 'Versicolor':
                    int_class.append(2)
                if i == 'Virginica':
                    int_class.append(3)

            df['int_class'] = int_class

            st.dataframe(df)

            # defining feature matrix(X) and response vector(y)
            x = df[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']]
            y = df['int_class']

            # y = y[:-2]
            print(x)
            print(y)

            reg = linear_model.LinearRegression()
            reg.fit(x, y)

            coefficient = reg.coef_
            intercept = reg.intercept_

            print(coefficient)
            print(intercept)


            # plt.scatter(x, y)
            # plt.xlabel('x')
            # plt.ylabel('y')
            # plt.show()
            # plot = px.scatter(
            #     df['sepal.length'], df['variety'], x='x', y='y')
            # st.ploty_chart(plot)

            # plot = px.scatter(x, y, x='x', y='y')
            # st.ploty_chart(plot)
            # reg.predict()

        if rad5 == "k-NN classifier":

            arr = []
            arr = df["variety"].unique()

            x = df.iloc[:, [0, 1, 2, 3]].values
            y = df.iloc[:, 4].values

            # Splitting the dataset into training and test set.

            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=0.25, random_state=0)

            # feature Scaling

            st_x = StandardScaler()
            x_train = st_x.fit_transform(x_train)
            x_test = st_x.transform(x_test)

            print(y_train)
            print("train")
            print(y_test)

            point = x_train[0]
            distance_points = []
            print(np.linalg.norm(point - x_train[2]))
            j = 0
            for i in range(len(x_train)):
                # distance_points[i] = np.linalg.norm(point - x_train[j])
                temp = point - x_train[i]
                sum = np.dot(temp.T, temp)
                distance_points.append(np.sqrt(sum))

            for i in range(len(x_train), len(df)):
                distance_points.append(1000)

            classifier = KNeighborsClassifier(
                n_neighbors=5, metric='minkowski', p=2)
            classifier.fit(x_train, y_train)

            y_pred = classifier.predict(x_test)

            m = confusion_matrix(y_test, y_pred)

            # print(point)
            # print(x_train[2])

            df["distance"] = distance_points

            x = df.iloc[:, [0, 1, 2, 3, 5]].values
            y = df.iloc[:, 4].values

            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=0.25)

            st_x = StandardScaler()
            x_train = st_x.fit_transform(x_train)
            x_test = st_x.transform(x_test)

            df = df.sort_values(by=['distance'])
            # print(df1)
            # st.subheader("After sorting")
            # st.dataframe(df)

            k_value = st.selectbox("k-value", [1, 3, 5, 7])

            df_first_k = df.head(k_value+1)

            st.dataframe(df_first_k)

            nearest_neighbour = (df_first_k['variety']).mode()
            st.subheader("Nearest " + str(k-values) +
                         "neighbours ", nearest_neighbour)
