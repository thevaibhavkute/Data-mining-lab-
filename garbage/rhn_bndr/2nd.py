from logging import critical
import pandas as pd
import scipy.stats as stats
import streamlit as st
import numpy as np
import plotly.express as px


st.title('DM Assignment-2')

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
# st.table(contingency)
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
# st.plotly_chart(df.plot(kind='bar'))
x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
y_values = st.sidebar.selectbox('Y axis', options=numeric_columns)
plot = px.scatter(data_frame=df, x=x_values, y=y_values)
# display chart
st.plotly_chart(plot)
# copy the data
df_min_max_scaled = df.copy()

stats.zscore(df_min_max_scaled)


# view normalized data
st.plotly_chart(df_min_max_scaled)
