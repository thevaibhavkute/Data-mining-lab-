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

def cal_median(data):
    n = len(data)
    data = list(data)
    data.sort()
    Median = -1
    if len(data) % 2 == 1:
        Median = float(data[int(n / 2)])
    else:
        pos1 = int(n / 2)
        pos2 = int((n - 1) / 2)
        Median = (data[pos1] + data[pos2]) / 2
    return Median

def statisticalDescription(dataList, attribute):
    result = ""

    Mean = 0.0
    data = dataList
    n = len(data)
    for val in data:
        Mean = Mean + val
    Mean /= n
    print("Mean:{}".format(Mean))
    result += "Mean:{}".format(Mean) + "\n"

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
        print("Frequency of all values is 1 , So there is no mode in this case")
        result += "Frequency of all values is 1 , So there is no mode in this case" + "\n"
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



def selectOneAttribute(numericDf, plotMehthod):
    print("Inside one attribute")

    columnList = list(numericDf.columns)

    menu = tk.Tk()
    menu.geometry("600x300")
    menu.title("Attribute Menu")

    j = 1
    i = 1
    for col in columnList:
        btn = Button(menu, text=col, command=partial(plotMehthod, numericDf, attribute=col))
        btn.grid(row=i, column=j)

        j += 1
        if j > 5:
            i += 1
            j = 1

    menu.mainloop()

def selectTwoAttribute(numericDf, plotMehthod):
    root = Tk()
    root.geometry('180x200')

    label = Label(root, text="Select 2 options")
    label.pack()

    listbox = Listbox(root, width=40, height=10, selectmode=MULTIPLE)
    
    columnList = list(numericDf.columns)
    
    i=0
    for col in columnList:
        listbox.insert(i+1,col)
        i += 1

    def selected_item():
        col=[]
        for i in listbox.curselection():
            col.append(listbox.get(i))
        print(col)
        plotMehthod(numericDf,col[0],col[1])

    btn = Button(root, text='Build Graph', command=selected_item)

    btn.pack(side='bottom')
    listbox.pack()

    root.mainloop()




def quantilePlot(numericDf, attribute):
    Y = list(numericDf[attribute])
    X = list()

    i = 1
    n = len(Y)
    for val in Y:
        X.append((i - 0.5) / n)
        i += 1
    X.sort()
    Y.sort()
    plt.scatter(X, Y)
    plt.show()

def qqPlot(numericDf, attribute):
    Y = list(numericDf[attribute])
    X = list()

    i = 1
    n = len(Y)
    for val in Y:
        X.append((i - 0.5) / n)
        i += 1
    X.sort()
    Y.sort()
    plt.scatter(X, Y)
    plt.show()

def scatterPlot(numericDf, x, y):
    plt.scatter(numericDf[x], numericDf[y], c = "blue")
    plt.show()

def histogramPlot(numericDf, attribute):
    print("Inside histogramPlot")

    plt.hist(numericDf[attribute])
    plt.show()

def boxplot(numericDf):
    plt.boxplot(numericDf)
    plt.show()




def graphicalAnalysis(numericDf):
    columnList = list(numericDf.columns)

    menu = tk.Tk()
    menu.geometry("300x300")
    menu.title("Attribute Menu")

    quantile = Button(menu, text="Quantile", command=partial(selectOneAttribute, numericDf, plotMehthod = quantilePlot))
    quantile.grid(row = 1, column = 1)

    quantile = Button(menu, text="Quantile - Quantile", command = partial(selectOneAttribute, numericDf, plotMehthod = qqPlot))
    quantile.grid(row = 2, column = 1)

    histogram = Button(menu, text="Histogram",command=partial(selectOneAttribute, numericDf, plotMehthod = histogramPlot))
    histogram.grid(row = 3, column = 1)

    scatter = Button(menu, text="Scatter", command=partial(selectTwoAttribute,numericDf,plotMehthod = scatterPlot))
    scatter.grid(row = 4, column = 1)

    box = Button(menu, text="Boxplot", command=partial(boxplot, numericDf))
    box.grid(row = 5, column = 1)

    menu.mainloop()

def numericalMenu(numericDf):
    columnList = list(numericDf.columns)

    menu = tk.Tk()
    menu.geometry("600x300")
    menu.title("Attribute Menu")

    j = 1
    i = 1
    for col in columnList:
        btn = Button(menu, text=col, command=partial(statisticalDescription, dataList=numericDf[col], attribute=col))
        btn.grid(row=i, column=j)

        j += 1
        if j > 5:
            i += 1
            j = 1

    menu.mainloop()

def browseFiles():
    filePath = filedialog.askopenfilename(initialdir="/", title="Select a File",filetypes=(("all files", "*.*"), ("all files", "*.*")))
    print("File Path: " + filePath)

    typeOfAnalysis(filePath)

def explorationWindow():
    window = Tk()

    window.title('File Explorer')
    label_file_explorer = Label(window, text="Choose dataset", width=100, height=4, fg="green")

    button_explore = Button(window, text="Browse Files", command=browseFiles)
    button_exit = Button(window, text="Exit", command=exit)
    label_file_explorer.grid(column=1, row=1)
    button_explore.grid(column=1, row=2)
    button_exit.grid(column=1, row=3)

    window.mainloop()

def typeOfAnalysis(filePath):

    df = pd.read_csv(filePath)
    numericDf = df.select_dtypes(include=np.number)

    window = Tk()
    window.title('Type Of  Analysis')
    label_file_explorer = Label(window, text="", width=100, height=4)

    numerical = Button(window, text="Numerical Analysis", command=partial(numericalMenu, numericDf))
    graphical = Button(window, text="Graphical Analysis", command=partial(graphicalAnalysis, numericDf))

    label_file_explorer.grid(column=1, row=1)
    numerical.grid(column=1, row=1)
    graphical.grid(column=1, row=2)

    window.mainloop()


if __name__ == "__main__":
    explorationWindow()
