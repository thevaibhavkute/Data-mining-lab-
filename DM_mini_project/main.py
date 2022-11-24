import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
st.set_option('deprecation.showPyplotGlobalUse', False)
# Import necessary libraries
import numpy as np
# import pandas as pd
import seaborn as sns


page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQxxRTtMZsy0XX07nREDgnG6_hJb7NqdSyXHz7MwTb_wY0Wb06dz3d3t44GpWlMUc_B5c4&usqp=CAU");
background-size: 100%;
background-position: top left;
background-repeat: true;
background-attachment: local;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

#variables
disrad = False

st.image("https://images.unsplash.com/photo-1501339847302-ac426a4a7cbb?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxzZWFyY2h8MTJ8fGNhZmV8ZW58MHx8MHx8&auto=format&fit=crop&w=500&q=60",width=800)

st.header("Cafe related Trends")

st.header(" Data Mining Mini project ")
st.write("1. 2019BTECS00067 Vaibhav Kute")
st.write("2. 2019BTECS00068 Aditi Joshi")
st.write("3. 2019BTECS00072 Nikhil Khavanekar")
st.write("4. 2019BTECS00073 Abhishek Kamble")

file_meta_data = st.file_uploader("Upload transactions here", type=['csv'], accept_multiple_files=False,disabled=False)

df = pd.read_csv(file_meta_data)
st.subheader("Transaction related trends")
# st.dataframe(df, width=2000, height=500)



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

trends = st.selectbox('Select the trend :',('Categories', 'Sell Catogory and Quantity sold', 'Catogory and Price', 'Qauntity sold and price', 'Sell id and quantity', 'Sell id and price','Price elasticity'))
# st.write('You selected:', Host_Country)



# if trends == 'Categories':
#
#
# elif trends == "Sell Catogory and Quantity sold":
# elif trends == "Catogory and Price":
# elif trends == "Qauntity sold and price":
# elif trends == "Sell id and quantity":
# elif trends == "Sell id and price":
# elif trends == "Price elasticity":
# elif trends == "Sell Catogory and Quantity sold":
# elif trends == "Sell Catogory and Quantity sold":
# elif trends == "Sell Catogory and Quantity sold":
# else:
# 	print("Please choose correct answer")


if trends == 'Categories':
    st.subheader("Categories")
    count_test = df['SELL_CATEGORY'].value_counts()
    labels = df['SELL_CATEGORY'].value_counts().index
    fig = plt.figure(figsize=(6, 6))
    plt.pie(count_test, labels=labels, autopct='%1.1f%%')
    plt.legend(labels)
    plt.show()
    st.pyplot(fig)

elif trends == "Sell Catogory and Quantity sold":
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




elif trends == "Catogory and Price":

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

elif trends == "Qauntity sold and price":

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

elif trends == "Sell id and quantity":

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

elif trends == "Sell id and price":

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
elif trends == "Price elasticity":

    st.subheader("Price elasticity")

    data=df

    data.sort_values(by=['SELL_ID'], inplace = True)

    data["% Change in Demand"] = df["QUANTITY"].pct_change()
    data["% Change in Price"] = df["PRICE"].pct_change()
    data["Price Elasticity"] = data["% Change in Demand"] / data["% Change in Price"]
    # print(data)
    st.write(data)

# print(data)
else:
	st.header("Please choose correct answer")


###########################################################################

file_holidays = st.file_uploader("Upload date info here", type=['csv'], accept_multiple_files=False,disabled=False)

df = pd.read_csv(file_holidays)
st.subheader("Date info related trend")
# st.dataframe(df, width=2000, height=500)


trends = st.selectbox('Select the trend related to date info:',('HOLIDAYs', 'Year and Temperature', 'School Breaks', 'Is outdoor and temperature', 'Is Outdoor', 'Holiday and average temperature','Is Weekends'))
# st.write('You selected:', Host_Country)
#
# if trends == 'HOLIDAYs':
# elif trends == "Year and Temperature":
# elif trends == "School Breaks":
# elif trends == "Is outdoor and temperature":
# elif trends == "Is Outdoor":
# elif trends == "Holiday and average temperature":
# elif trends == "Is Weekends":
# elif trends == "Sell Catogory and Quantity sold":
# elif trends == "Sell Catogory and Quantity sold":
# elif trends == "Sell Catogory and Quantity sold":
# else:
# 	print("Please choose correct answer")


if trends == 'HOLIDAYs':
    st.subheader("HOLIDAYs")
    count_test = df['HOLIDAY'].value_counts()
    labels = df['HOLIDAY'].value_counts().index
    fig = plt.figure(figsize=(6, 6))
    plt.pie(count_test, labels=labels, autopct='%1.1f%%')
    plt.legend(labels)
    plt.show()
    st.pyplot(fig)



elif trends == "Year and Temperature":

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


elif trends == "School Breaks":


    st.subheader("School Breaks")
    count_test = df['IS_SCHOOLBREAK'].value_counts()
    labels = df['IS_SCHOOLBREAK'].value_counts().index
    fig = plt.figure(figsize=(6, 6))
    plt.pie(count_test, labels=labels, autopct='%1.1f%%')
    plt.legend(labels)
    plt.show()
    st.pyplot(fig)

elif trends == "Is outdoor and temperature":


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


elif trends == "Is Outdoor":


    st.subheader("Outdoor")
    count_test = df['IS_OUTDOOR'].value_counts()
    labels = df['IS_OUTDOOR'].value_counts().index
    fig = plt.figure(figsize=(6, 6))
    plt.pie(count_test, labels=labels, autopct='%1.1f%%')
    plt.legend(labels)
    plt.show()
    st.pyplot(fig)

elif trends == "Holiday and average temperature":

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


elif trends == "Is Weekends":

    st.subheader("Weekends")

    count_test = df['IS_WEEKEND'].value_counts()
    labels = df['IS_WEEKEND'].value_counts().index
    fig = plt.figure(figsize=(6, 6))
    plt.pie(count_test, labels=labels, autopct='%1.1f%%')
    plt.legend(labels)
    plt.show()
    st.pyplot(fig)
else :
    st.write("Please choose correct answer")
######################################################

