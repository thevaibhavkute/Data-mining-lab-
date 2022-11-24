import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
st.set_option('deprecation.showPyplotGlobalUse', False)
# Import necessary libraries
import numpy as np
# import pandas as pd
import seaborn as sns

st.header(" Data Mining Mini project ")
st.write("1. 2019BTECS00067 Vaibhav Kute")
st.write("2. 2019BTECS00068 Aditi Joshi")
st.write("3. 2019BTECS00072 Nikhil Khavanekar")
st.write("4. 2019BTECS00073 Abhishek Kamble")
#
file_meta_data = st.file_uploader("Upload transactions here", type=['csv'], accept_multiple_files=False,disabled=False)

df = pd.read_csv(file_meta_data)
st.subheader("Dataset loaded Table")
st.dataframe(df, width=2000, height=500)

# file_transactions = st.file_uploader("Upload transactions here", type=['csv'], accept_multiple_files=False,disabled=False)
#
# df = pd.read_csv(file_transactions)
# st.subheader("Dataset loaded Table")
# st.dataframe(df, width=2000, height=500)
#
# file_holidays = st.file_uploader("Upload date info here", type=['csv'], accept_multiple_files=False,disabled=False)
#
# df = pd.read_csv(file_holidays)
# st.subheader("Dataset loaded Table")
# st.dataframe(df, width=2000, height=500)

######################################################################



st.subheader("Categories")
count_test = df['SELL_CATEGORY'].value_counts()
labels = df['SELL_CATEGORY'].value_counts().index
fig = plt.figure(figsize=(6, 6))
plt.pie(count_test, labels=labels, autopct='%1.1f%%')
plt.legend(labels)
plt.show()
st.pyplot(fig)


#1
st.subheader("Sell Catogory and Quantity sold")

atr1, atr2 = st.columns(2)
attribute1 = df.columns[-3]
# atr1.selectbox("Select Attribute 1",df.columns[-2])
# attribute2 = atr2.selectbox("Select Attribute 2", cols)
# printf(attribute1)
# printf(attribute2)
classatr = df.columns[-1]
sns.set_style("whitegrid")
sns.FacetGrid(df, hue=classatr, height=5).map(sns.histplot, attribute1).add_legend()
plt.title("Sell Catogory and Quantity sold")
plt.show(block=True)
st.pyplot()

#2
st.subheader("Catogory and Price")

atr1, atr2 = st.columns(2)
attribute1 = df.columns[-4]
# atr1.selectbox("Select Attribute 1",df.columns[-2])
# attribute2 = atr2.selectbox("Select Attribute 2", cols)
# printf(attribute1)
# printf(attribute2)
classatr = df.columns[-1]
sns.set_style("whitegrid")
sns.FacetGrid(df, hue=classatr, height=5).map(sns.histplot, attribute1).add_legend()
plt.title("Catogory and Price")
plt.show(block=True)
st.pyplot()

#3
st.subheader("Qauntity sold and price")

atr1, atr2 = st.columns(2)
attribute1 = df.columns[-3]
# atr1.selectbox("Select Attribute 1",df.columns[-2])
# attribute2 = atr2.selectbox("Select Attribute 2", cols)
# printf(attribute1)
# printf(attribute2)
classatr = df.columns[-4]
sns.set_style("whitegrid")
sns.FacetGrid(df, hue=classatr, height=5).map(sns.histplot, attribute1).add_legend()
plt.title("Qauntity sold and price")
plt.show(block=True)
st.pyplot()

#4
st.subheader("Sell id and quantity")

atr1, atr2 = st.columns(2)
attribute1 = df.columns[-3]
# atr1.selectbox("Select Attribute 1",df.columns[-2])
# attribute2 = atr2.selectbox("Select Attribute 2", cols)
# printf(attribute1)
# printf(attribute2)
classatr = df.columns[-2]
sns.set_style("whitegrid")
sns.FacetGrid(df, hue=classatr, height=5).map(sns.histplot, attribute1).add_legend()
plt.title("Sell id and quantity")
plt.show(block=True)
st.pyplot()


#5
st.subheader("Sell id and price")

atr1, atr2 = st.columns(2)
attribute1 = df.columns[-4]
# atr1.selectbox("Select Attribute 1",df.columns[-2])
# attribute2 = atr2.selectbox("Select Attribute 2", cols)
# printf(attribute1)
# printf(attribute2)
classatr = df.columns[-2]
sns.set_style("whitegrid")
sns.FacetGrid(df, hue=classatr, height=5).map(sns.histplot, attribute1).add_legend()
plt.title("Sell id and price")
plt.show(block=True)
st.pyplot()

###########################################################################

file_holidays = st.file_uploader("Upload date info here", type=['csv'], accept_multiple_files=False,disabled=False)

df = pd.read_csv(file_holidays)
st.subheader("Dataset loaded Table")
st.dataframe(df, width=2000, height=500)





st.subheader("HOLIDAYs")
count_test = df['HOLIDAY'].value_counts()
labels = df['HOLIDAY'].value_counts().index
fig = plt.figure(figsize=(6, 6))
plt.pie(count_test, labels=labels, autopct='%1.1f%%')
plt.legend(labels)
plt.show()
st.pyplot(fig)




#1
st.subheader("Year and Temperature")

atr1, atr2 = st.columns(2)
attribute1 = df.columns[-2]
# atr1.selectbox("Select Attribute 1",df.columns[-2])
# attribute2 = atr2.selectbox("Select Attribute 2", cols)
# printf(attribute1)
# printf(attribute2)
classatr = df.columns[1]
sns.set_style("whitegrid")
sns.FacetGrid(df, hue=classatr, height=5).map(sns.histplot, attribute1).add_legend()
plt.title("Year and Temperature")
plt.show(block=True)
st.pyplot()


st.subheader("School Breaks")
count_test = df['IS_SCHOOLBREAK'].value_counts()
labels = df['IS_SCHOOLBREAK'].value_counts().index
fig = plt.figure(figsize=(6, 6))
plt.pie(count_test, labels=labels, autopct='%1.1f%%')
plt.legend(labels)
plt.show()
st.pyplot(fig)



#2
st.subheader("Is outdoor and temperature")

atr1, atr2 = st.columns(2)
attribute1 = df.columns[-2]
# atr1.selectbox("Select Attribute 1",df.columns[-2])
# attribute2 = atr2.selectbox("Select Attribute 2", cols)
# printf(attribute1)
# printf(attribute2)
classatr = df.columns[-1]
sns.set_style("whitegrid")
sns.FacetGrid(df, hue=classatr, height=5).map(sns.histplot, attribute1).add_legend()
plt.title("Is outdoor and temperature")
plt.show(block=True)
st.pyplot()

st.subheader("Outdoor")
count_test = df['IS_OUTDOOR'].value_counts()
labels = df['IS_OUTDOOR'].value_counts().index
fig = plt.figure(figsize=(6, 6))
plt.pie(count_test, labels=labels, autopct='%1.1f%%')
plt.legend(labels)
plt.show()
st.pyplot(fig)


#3
st.subheader("Holiday and average temperature")

atr1, atr2 = st.columns(2)
attribute1 = df.columns[-2]
# atr1.selectbox("Select Attribute 1",df.columns[-2])
# attribute2 = atr2.selectbox("Select Attribute 2", cols)
# printf(attribute1)
# printf(attribute2)
classatr = df.columns[2]
sns.set_style("whitegrid")
sns.FacetGrid(df, hue=classatr, height=5).map(sns.histplot, attribute1).add_legend()
plt.title("Holiday and average temperature")
plt.show(block=True)
st.pyplot()


st.subheader("Weekends")

count_test = df['IS_WEEKEND'].value_counts()
labels = df['IS_WEEKEND'].value_counts().index
fig = plt.figure(figsize=(6, 6))
plt.pie(count_test, labels=labels, autopct='%1.1f%%')
plt.legend(labels)
plt.show()
st.pyplot(fig)


######################################################

