from json import load
from xmlrpc.client import Boolean
import streamlit as st
import pandas as pd
import numpy as np
import time 
import matplotlib.pyplot as plt
from apps import MultiApp
from Apps import asg1, asg2, asg3, asg5, asg6, asg7, asg8

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://stoque.com.br/wp-content/uploads/2022/02/data_mining.png");
background-size: 180%;
background-position: top left;
background-repeat: true;
background-attachment: local;
}}

</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)
#variables
disrad = False

st.image("https://www.netsuite.com/portal/assets/img/business-articles/data-warehouse/social-data-mining.jpg?v2",width=800)
# st.image("https://artoftesting.com/wp-content/uploads/2022/02/data-mining.png",width=800)

st.subheader("Name: Vaibhav Kute {2019BTECS00067}")
st.write("Under Guidance: Dr.BFM")
# file = st.file_uploader("Upload dataset here", type=['csv'], accept_multiple_files=False,disabled=False)
# data = pd.read_csv(file)

# def load_file():
#     df = pd.read_csv(file)
#     st.subheader("Dataset loaded Table")
#     st.dataframe(df, width=1000, height=500)
#     return df
# if file:
#     data = load_file()
#
app = MultiApp()

app.add_app("Assignment No.1", asg1.app)
app.add_app("Assignment No.2", asg2.app)
app.add_app("Assignment No.3 and 4", asg3.app)
app.add_app("Assignment No.5", asg5.app)
app.add_app("Assignment No. 6",asg6.app)
app.add_app("Assignment No. 7",asg7.app)
app.add_app("Assignment No. 8",asg8)

# The main app
app.run()

