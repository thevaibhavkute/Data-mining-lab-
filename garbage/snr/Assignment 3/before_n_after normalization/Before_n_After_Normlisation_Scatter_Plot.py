#!/usr/bin/env python
# coding: utf-8

# In[63]:


from tkinter import *
import tkinter as tk
import tkinter.ttk as ttk
import csv
from tkinter import filedialog
from functools import partial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy.stats
from pandastable import Table,TableModel
import os


# In[72]:



def z_score(x,col):
 n=len(col)
 Mean = sum(x)/len(x)
 #     variance calculation
 Variance = 0.0
 for val in x:
     Variance += (val-Mean)*(val-Mean)/(n-1)

 #     Standard Deviation
 StandardDeviation = math.sqrt(Variance)

 z_score = []

 for val in x:
     z_score.append((val-Mean)/StandardDeviation)
 # print(z_score)
 return z_score


# In[75]:



def min_max(x,col):
    # Min_Max_Normalization 
    # starts here
    n=len(x)
    xmin = min(x)
    xmax = max(x)
    lmin = 0  # local min
    lmax = 1  # local max

    min_max = []
    if xmin == xmax:
        print("denominator became zero because min and max are same")
    else:
        
        for val in x:
            min_max.append((val-xmin)/(xmax-xmin)*(lmax-lmin)+lmin)

        # print(min_max)
        x=list(x)
    return min_max


# In[66]:



def decimal(x,col):
    #     Decimal_Scaling
    x = list(x)
    n = len(x)
    denom = pow(10, len(str(max(x))))

    decimal_scaling = []
    for val in x:
        decimal_scaling.append(val/denom)

    return decimal_scaling


# In[76]:


print("Change The filepath variable first")

filePath = "D:/abhishek/7th SEM/DM Lab/Assignment 3/iris.csv"

df = pd.read_csv(filePath)
# print(df)
print("This code is for before and after normalization scatterplot ")


print("before normalization")
x=list(df['sepal length'])
y=list(df['sepal width'])
# x.sort()
# y.sort()
plt.scatter(x, y)
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.show()


x=z_score(df['sepal length'],'sepal length')
y=z_score(df['sepal width'],'sepal width')

print("after z_score normalization scatter plot is as follows")
plt.xlabel('sepal length')
plt.ylabel('sepal width')
# x.sort()
# y.sort()
plt.scatter(x, y)
plt.show()

x=min_max(df['sepal length'],'sepal length')
y=min_max(df['sepal width'],'sepal width')

print("after min-max normalization scatter plot is as follows")
plt.xlabel('sepal length')
plt.ylabel('sepal width')
# x.sort()
# y.sort()
plt.scatter(x, y)
plt.show()

x=decimal(df['sepal length'],'sepal length')
y=decimal(df['sepal width'],'sepal width')

print("after decimal normalization scatter plot is as follows")
plt.xlabel('sepal length')
plt.ylabel('sepal width')
# x.sort()
# y.sort()
plt.scatter(x, y)
plt.show()


# In[ ]:




