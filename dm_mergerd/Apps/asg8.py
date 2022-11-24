import requests
from bs4 import BeautifulSoup
import streamlit as st


def app(data):
    st.header("Assignment No. 8")
    st.write("2019BTECS00067 - Vaibhav Kute")

    # def app(data):
    #     urls = 'https://www.geeksforgeeks.org/'
    title = st.text_input('Enter the link', 'Enter link here')
    # urls=input('Please enter the url: ')
    grab = requests.get(title)
    soup = BeautifulSoup(grab.text, 'html.parser')

    # opening a file in write mode
    # f = open("test1.txt", "w")
    # traverse paragraphs from soup
    for link in soup.find_all("a"):
        data = link.get('href')
        st.write(data)
