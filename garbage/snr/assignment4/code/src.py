from tkinter import *
import tkinter as tk
from tkinter import messagebox
from tkinter import commondialog
from tkinter import scrolledtext
from pandastable import Table, TableModel
import pandas as pd
import os
from tkinter import filedialog
from functools import partial
import csv
from tkintertable.Tables import TableCanvas
from tkintertable.TableModels import TableModel
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn import preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as img


Target = "Unassigned"
clf = DecisionTreeClassifier()
(X_train, X_test, y_train, y_test) = [],[],[],[] 
y_pred = []
df = pd.DataFrame()

def browseFiles():
	global df
	filePath = filedialog.askopenfilename(initialdir="/", title="Select a File",filetypes=(("all files", "*.*"), ("all files", "*.*")))
	df =  pd.read_csv(filePath)
	refresh(df)

def freshStart():
	global df
	df = pd.DataFrame(index = ["NOTE:", "", ""], data = ["Upload a Dataset and"," select the target ","attribute"])
	refresh(df)

def setTarget(col):
	global Target
	Target = col
	refresh(df)

def buildTree(method):
	global df
	global Target
	global clf, X_train, X_test, y_train, y_test, y_pred

	le = preprocessing.LabelEncoder()
	for column_name in df.columns:
		if df[column_name].dtype == object:
			df[column_name] = le.fit_transform(df[column_name])
		else:
			pass

	inputList = list(df.columns)
	inputList.remove(Target)

	df[Target] = list(map(str, df[Target]))

	X = df[inputList].values
	y = df[Target].values


	(X_train, X_test, y_train, y_test) = train_test_split(X, y, train_size = 0.7, random_state = 1)
	clf = DecisionTreeClassifier(criterion = method, random_state = 90, max_depth = 4)
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)

	refresh(df)

def showConfusionMatrix(y_test, y_pred):
	tmp_df = pd.DataFrame(data = confusion_matrix(y_test, y_pred))
	refresh(tmp_df)

def performance_metrics():
	global clf, X_train, X_test, y_train, y_test, y_pred
	inputList = list(df.columns)
	inputList.remove(Target)

	report = classification_report(y_test, y_pred, output_dict=True)
	new_df = pd.DataFrame(report).transpose()

	refresh(new_df)

def displayTree():
	feature_cols = list(df.columns)
	feature_cols.remove(Target)

	
	tree.plot_tree(clf, 
          feature_names = feature_cols, 
          class_names = list(set(df[Target])), 
          filled = True, 
          rounded = True,
		  fontsize=7)
	
	plt.show()

def appWindow():
	global df
	global root
	global splitting_method
	global Target
	global clf, X_train, X_test, y_train, y_test, y_pred

	root = Tk()
	root.title("Home page")
	root['background'] = '#9ea7e6'
	width= root.winfo_screenwidth() 
	height= root.winfo_screenheight()
	root.geometry("%dx%d" % (width, height))
	root.state('zoomed')

	# Creating Menubar
	menubar = Menu(root)
	  
	Dataset = Menu(menubar, tearoff = 0)
	menubar.add_cascade(label ='Dataset', menu = Dataset)
	Dataset.add_command(label ='Upload', command = browseFiles)
	Dataset.add_command(label ='Load Dataset', command = partial(refresh, df))
	Dataset.add_command(label ='Delete', command = partial(refresh, pd.DataFrame(data = [""])))
	
	Target_Attribute = Menu(Dataset, tearoff = 0)
	Dataset.add_cascade(label='Target Attribute', menu = Target_Attribute)
	for col in df.columns:
		Target_Attribute.add_command(label = col, command = partial(setTarget, col))

	Dataset.add_separator()
	Dataset.add_command(label ='Exit', command = root.destroy)
	
	
	Decision_Tree = Menu(menubar, tearoff = 0)
	menubar.add_cascade(label ='Decision_Tree', menu = Decision_Tree)


	Build = Menu(Decision_Tree, tearoff = 0)
	Decision_Tree.add_cascade(label='Build', menu=Build)
	Build.add_command(label = 'Gini Index', command = partial(buildTree, method = "gini"))
	Build.add_command(label = 'Gain Ratio', command = None)
	Build.add_command(label = 'Information Gain', command = partial(buildTree, method = "entropy"))

	Decision_Tree.add_command(label = 'Display Tree', command = displayTree)


	Performance_Evaluation = Menu(menubar, tearoff = 0)
	menubar.add_cascade(label ='Performance_Evaluation', menu = Performance_Evaluation)
	Performance_Evaluation.add_command(label ='Confusion Matrix',\
		command = partial(showConfusionMatrix, y_test, y_pred))
	Performance_Evaluation.add_command(label ='Performance Metrics', command = performance_metrics)
	  
	# display Menu
	root.config(menu = menubar)

	tframe = Frame(root, bg = '#9ea7e6')
	tframe.pack(expand = True, fill = BOTH)
	table = TableCanvas(tframe)
	table.importCSV("util.csv", sep=',')
	table.show()

	root.mainloop()


def refresh(new_df):
	global root

	try:
		root.destroy()
	except:
		pass

	new_df.to_csv("util.csv")
	appWindow()


if __name__ == "__main__":
	freshStart()
