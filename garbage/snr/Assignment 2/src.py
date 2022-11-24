from tkinter import *
import tkinter as tk
import tkinter.ttk as ttk
import csv
from tkinter import filedialog
from functools import partial
import numpy as np
import pandas as pd
import math

def cal_median(data):
    n=len(data)
    # print("length is {}".format(n))
    data=list(data)
    data.sort()
    Median=-1
    if len(data)%2==1:
        Median=float(data[int(n/2)])
    else:
        pos1=int(n/2)
        pos2=int((n-1)/2)
        Median=(data[pos1]+data[pos2])/2
    return Median
	
def statisticalDescription(dataList, attribute):
	result = "Displaying values for '" + attribute +"' => \n\n"

	Mean = 0.0
	data=dataList
	n = len(data)
	for val in data:
		Mean = Mean + val
	Mean /= n
	print("Mean:{}".format(Mean))
	result += "Mean:{}".format(Mean) + "\n"
	#     print(data.mean())

	#     median calculation
	data = list(data)
	data.sort()
	Median = -1
	if len(data) % 2 == 1:
		Median = float(data[int(n / 2)])
	else:
		pos1 = int(n / 2)
		pos2 = int((n - 1) / 2)
		Median = (data[pos1] + data[pos2]) / 2
	print("Median:{}".format(Median))
	result += "Median:{}".format(Median) + "\n"
	#     Mode calculation
	tmp = {}
	for val in data:
		if val in tmp:
			tmp[val] = tmp[val] + 1
		else:
			tmp[val] = 1
	max_frequency = max(tmp.values())
	Modes = []
	#     print(tmp)
	for val in tmp.keys():
		if tmp[val] == max_frequency:
			Modes.append(val)
	if len(Modes) == len(data):
		print("Frequency of all values is same , So there is no mode in this case")
		result += "Frequency of all values is same , So there is no mode in this case"  + "\n"
	else:
		print("Modes are as follows: ", end=" ")
		print(Modes)
		result += "Modes are as follows: "
		for val in Modes:
			result += str(val) + " "
		# check here later
		result += "\n"

	#     Midrange
	print("Midrange:{}".format((min(data) + max(data)) / 2))
	result += "Midrange:{}".format((min(data) + max(data)) / 2) + "\n"

	#     variance calculation
	Variance = 0.0
	for val in data:
		Variance += (val - Mean) * (val - Mean) / (n - 1)
	print("Variance:{}".format(Variance))
	result += "Variance:{}".format(Variance) + "\n"

	#     Standard Deviation
	StandardDeviation = math.sqrt(Variance)
	print("StandardDeviation:{}".format(StandardDeviation))
	result += "StandardDeviation:{}".format(StandardDeviation) + "\n"

	#     Range
	Range = max(data) - min(data)
	print("Range:{}".format(Range))
	result += "Range:{}".format(Range) + "\n"

	#     quartiles
	Quartiles = []
	Quartiles.append(cal_median(data))  # mid Quartiles
	Quartiles.append(cal_median(data[:int((n + 1) / 2)]))  # upper Quartiles
	Quartiles.append(cal_median(data[int((n + 1) / 2):]))  # lower Quartiles
	Quartiles.sort()
	print("Quartiles: ", end=" ")
	print(Quartiles)
	result += "Quartiles: " + "\n"
	for val in Quartiles:
		result += str(val) + " "
	result += "\n"

	#     interquartile range
	Interquartile_Range = max(Quartiles) - min(Quartiles)
	print("Interquartile_Range:{} ".format(Interquartile_Range))
	result += "Interquartile_Range:{} ".format(Interquartile_Range) + "\n"

	#     five-number summary
	Five_Number_Summary = []
	Five_Number_Summary.append(min(data))  # min value
	for val in Quartiles:  # quartiles
		Five_Number_Summary.append(val)
	Five_Number_Summary.append(max(data))  # max value

	print("Five-Number Summary: ", end=" ")
	print(Five_Number_Summary)
	result += "Five-Number Summary: "
	for val in Five_Number_Summary:
		result += str(val) + " "
	result += "\n"

	root = tk.Tk()
	T = tk.Text(root, height=30, width=100)
	T.pack()
	T.insert(tk.END, result)
	tk.mainloop()


def attributeMenu(filePath):
	filePath = 'D:/abhishek/7th SEM/DM Lab/Assignment 2/breast-cancer-wisconsin.csv'

	df = pd.read_csv (filePath)
	print( df.head())
	numericDf = df.select_dtypes(include=np.number)

	columnList = numericDf

	menu = tk.Tk()
	menu.geometry("600x300")
	menu.title("Attribute Menu")

	var = 0
	for col in columnList:
		btn = Button(menu, text = col, command = partial(statisticalDescription, dataList = numericDf[col], attribute = col))
		btn.grid(row = 1,column = var)

		var += 1

	menu.mainloop()


def browseFiles():
	filePath = filedialog.askopenfilename(initialdir = "/",title = "Select a File",filetypes = (("all files","*.*"),("all files","*.*")))
	print("File Path: " + filePath)

	attributeMenu(filePath)

def explorationWindow():
	window = Tk()
	
	window.title('File Explorer')
	label_file_explorer = Label(window,text = "Choose dataset",width = 100, height = 4,fg = "green")

	button_explore = Button(window,text = "Browse Files",command = browseFiles)
	button_exit = Button(window,text = "Exit",command = exit)
	label_file_explorer.grid(column = 1, row = 1)
	button_explore.grid(column = 1, row = 2)
	button_exit.grid(column = 1,row = 3)

	window.mainloop()

def redundant():

	for i in range(100):
		print(i)


	# "mean"
	# "median"
	# "mode"
	# "midrange"
	# "variance"
	# "standard deviation"
	# "range"
	# "quartiles"
	# "interquartile range"
	# "five-number summary"


if __name__ == "__main__":
	#explorationWindow()
	attributeMenu("")

