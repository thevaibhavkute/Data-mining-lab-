
from collections import Counter
import math
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# setting header
st.header('Assignment No.1')

# getting file from  the user
uploaded_file = st.file_uploader(label="Choose a file", type=['csv', 'xlsx'])

# to store the data
global dataFrame
global numeric_columns

# after uploading of file
if uploaded_file is not None:
    # print(uploaded_file)
    print("File uploaded successfullly")
    try:
        dataFrame = pd.read_csv(uploaded_file)
    except Exception as e:
        print(e)
        dataFrame = pd.read_excel(uploaded_file)

try:
    st.write(dataFrame)
    numeric_columns = list(dataFrame.columns)
except Exception as e:
    print(e)
    st.write("Please upload file")

st.subheader("Central tendency and dispersion of data")

mean1 = dataFrame['sepal.length'].mean()

median1 = dataFrame['sepal.length'].median()
mode1 = dataFrame['sepal.length'].mode()
# midRange1 = dataFrame['Salary'].()

std1 = dataFrame['sepal.length'].std()
var1 = dataFrame['sepal.length'].var()
quantile1 = dataFrame['sepal.length'].quantile()

print('Mean salary: ' + str(mean1))

st.write("Mean : " + str(mean1))
st.write("Median : " + str(median1))
st.write("Mode : " + str(mode1))
# st.write("Midrange : " + str(midRange1))
st.write("variance : " + str(var1))
st.write("Standard Deviation : " + str(std1))
st.write("Quantile : " + str(quantile1))

print(dataFrame.columns.values)
print(dataFrame['sepal.length'].values)

# Calculate mean
mean1 = 0
count = 0
for x in dataFrame['sepal.length'].values:
    mean1 += x
    count = count + 1
mean1 = mean1/count
print(mean1)

# Calcualte Median

data1 = dataFrame['sepal.length'].values


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

# for i in range(len(data1)):
#     arr.append(data1.loc[i, attribute])
#     arr.sort()
# print("Midrange of given dataset is ("+attribute+") "+str((arr[n-1]+arr[0])/2))
# st.write("Midrange of given dataset is (" + attribute+") "+str((arr[n-1]+arr[0])/2))


def variance(data1):
    # Number of observations
    n = len(data1)
    # Mean of the data
    mean = sum(data1) / n
    # Square deviations
    deviations = [(x - mean) ** 2 for x in data1]
    # Variance
    variance = sum(deviations) / n
    return variance


print(variance(data1))


def stdev(data1):
    var = variance(data1)
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

rank(data1)

st.subheader("Graphical display of above calculated statistical description of data")
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
        plot = px.scatter(data_frame=dataFrame, x=x_values, y=y_values)
        # display chart
        st.plotly_chart(plot)
    except Exception as e:
        print(e)

if chart_select == 'Lineplots':
    st.sidebar.subheader("Lineplot Settings")
    try:
        x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
        y_values = st.sidebar.selectbox('Y axis', options=numeric_columns)
        plot = px.line(data_frame=dataFrame, x=x_values, y=y_values)
        # display chart
        st.plotly_chart(plot)
    except Exception as e:
        print(e)

if chart_select == 'Histogram':
    st.sidebar.subheader("Histogram Settings")
    try:
        x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
        y_values = st.sidebar.selectbox('Y axis', options=numeric_columns)
        plot = px.histogram(data_frame=dataFrame, x=x_values, y=y_values)
        # display chart
        st.plotly_chart(plot)
    except Exception as e:
        print(e)

if chart_select == 'Boxplot':
    st.sidebar.subheader("Boxplot Settings")
    try:
        x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
        y_values = st.sidebar.selectbox('Y axis', options=numeric_columns)
        plot = px.box(data_frame=dataFrame, x=x_values, y=y_values)
        # display chart
        st.plotly_chart(plot)
    except Exception as e:
        print(e)
